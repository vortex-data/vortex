// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Column-major decoding of shredded maps with string values.
//!
//! Each row's output offset is computed once. Columns are then visited in key order, and every
//! present row writes its key and value views straight into its next output slot. Residual
//! entries are bucketed by how many column keys sort before them, so they land between the right
//! columns. There is no per-row loop over absent columns, no string comparison per entry and no
//! intermediate list of value positions.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::decode::Remap;
use crate::decode::unwrap_layers;
use crate::flat::FlatMap;
use crate::flat::Strings;

/// Decoded keys and, optionally, values of a shredded map, with per-row offsets and sizes.
pub(crate) struct Columnar {
    pub offsets: Vec<u64>,
    pub sizes: Vec<u64>,
    pub keys: ArrayRef,
    pub values: Option<ArrayRef>,
}

fn for_each_set(mask: &Mask, mut f: impl FnMut(usize)) {
    match mask.bit_buffer() {
        AllOr::All => (0..mask.len()).for_each(f),
        AllOr::None => {}
        AllOr::Some(bits) => bits.for_each_set_index(&mut f),
    }
}

/// Collects views from several string arrays into one output, rebasing buffer indices.
struct ViewSink {
    buffers: Vec<ByteBuffer>,
}

impl ViewSink {
    /// Registers an array's buffers, returning the base index for its views.
    fn add(&mut self, array: &VarBinViewArray) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        let base = self.buffers.len() as u32;
        self.buffers
            .extend(array.data_buffers().iter().map(|b| b.as_host().clone()));
        base
    }
}

#[inline]
fn rebase(view: BinaryView, base: u32) -> BinaryView {
    if view.is_inlined() || base == 0 {
        view
    } else {
        let r = view.as_view();
        BinaryView::from(r.with_buffer_and_offset(r.buffer_index + base, r.offset))
    }
}

/// A shredded column resolved to its innermost string views.
struct Column {
    views: Vec<BinaryView>,
    base: u32,
    layers: Vec<Remap>,
    cursor: usize,
}

impl Column {
    /// The value view of `row`. Rows must be visited in ascending order.
    #[inline]
    fn view(&mut self, row: usize) -> BinaryView {
        let mut pos = row;
        let mut layers = self.layers.iter();
        if let Some(Remap::Rank(rows)) = self.layers.first() {
            while rows.get(self.cursor).is_some_and(|&r| r < pos) {
                self.cursor += 1;
            }
            pos = self.cursor;
            layers.next();
        }
        for layer in layers {
            pos = layer.apply(pos);
        }
        rebase(self.views[pos], self.base)
    }
}

/// Decodes `flat` (the residual) merged with string `columns`, keyed by `keys` in sorted order.
///
/// Rows set in `repeats` reuse the previous row's entries.
pub(crate) fn decode_columnar(
    flat: &FlatMap,
    keys: &[&str],
    column_arrays: &[ArrayRef],
    masks: &[Mask],
    repeats: &BitBuffer,
    with_values: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Columnar> {
    let len = flat.len;
    let residual_keys = Strings::new(&flat.keys);

    // Row sizes, then offsets. Repeated rows reuse their predecessor's range.
    let mut counts = vec![0u32; len];
    for row in 0..len {
        #[allow(clippy::cast_possible_truncation)]
        {
            counts[row] = flat.range(row).len() as u32;
        }
    }
    for mask in masks {
        for_each_set(mask, |row| counts[row] += 1);
    }
    let mut offsets = Vec::with_capacity(len);
    let mut sizes = Vec::with_capacity(len);
    let mut total = 0u64;
    for row in 0..len {
        if repeats.value(row) {
            offsets.push(offsets[row - 1]);
            sizes.push(sizes[row - 1]);
        } else {
            offsets.push(total);
            sizes.push(u64::from(counts[row]));
            total += u64::from(counts[row]);
        }
    }
    let total = usize::try_from(total)?;

    // Residual entries bucketed by the number of column keys sorting at or before them, so they
    // follow equal shredded keys and precede the next column.
    let mut buckets: Vec<Vec<(u32, u32)>> = vec![Vec::new(); keys.len() + 1];
    for row in 0..len {
        if repeats.value(row) {
            continue;
        }
        for j in flat.range(row) {
            let key = residual_keys.get(j);
            let slot = keys.partition_point(|k| k.as_bytes() <= key);
            #[allow(clippy::cast_possible_truncation)]
            buckets[slot].push((row as u32, j as u32));
        }
    }

    // Key views point into the residual key buffers and a pool of column key names.
    let mut key_sink = ViewSink {
        buffers: Vec::new(),
    };
    let residual_key_base = key_sink.add(&flat.keys);
    #[allow(clippy::cast_possible_truncation)]
    let pool_index = key_sink.buffers.len() as u32;
    let mut pool = Vec::new();
    let column_key_views: Vec<BinaryView> = keys
        .iter()
        .map(|key| {
            #[allow(clippy::cast_possible_truncation)]
            let offset = pool.len() as u32;
            pool.extend_from_slice(key.as_bytes());
            BinaryView::make_view(key.as_bytes(), pool_index, offset)
        })
        .collect();
    key_sink.buffers.push(ByteBuffer::from(pool));

    let mut value_sink = ViewSink {
        buffers: Vec::new(),
    };
    let (residual_values, residual_value_valid, residual_value_base, mut columns) = if with_values
    {
        let values = flat.values.clone().execute::<VarBinViewArray>(ctx)?;
        let valid = values.validity()?.execute_mask(values.len(), ctx)?;
        let base = value_sink.add(&values);
        let mut columns = Vec::with_capacity(column_arrays.len());
        for array in column_arrays {
            let (inner, layers) = unwrap_layers(array, ctx)?;
            let inner = inner.execute::<VarBinViewArray>(ctx)?;
            let base = value_sink.add(&inner);
            columns.push(Column {
                views: inner.views().to_vec(),
                base,
                layers,
                cursor: 0,
            });
        }
        (Some(values), Some(valid), base, columns)
    } else {
        (None, None, 0, Vec::new())
    };

    let mut key_views = vec![BinaryView::empty_view(); total];
    let mut value_views = if with_values {
        vec![BinaryView::empty_view(); total]
    } else {
        Vec::new()
    };
    let mut value_valid = BitBufferMut::new_set(if with_values { total } else { 0 });
    let mut fill = vec![0u32; len];
    let residual_views: Option<Vec<BinaryView>> = residual_values.as_ref().map(|v| v.views().to_vec());
    let residual_key_views: Vec<BinaryView> = residual_keys.views.to_vec();

    // Visiting every column over all rows would scatter writes across the whole output once per
    // column, so rows are processed in blocks whose output slice stays in cache.
    let present_rows: Vec<Vec<u32>> = masks
        .iter()
        .map(|mask| {
            let mut rows = Vec::with_capacity(mask.true_count());
            #[allow(clippy::cast_possible_truncation)]
            for_each_set(mask, |row| rows.push(row as u32));
            rows
        })
        .collect();
    let mut column_cursor = vec![0usize; keys.len()];
    let mut bucket_cursor = vec![0usize; keys.len() + 1];
    const BLOCK: usize = 1024;
    for block_start in (0..len).step_by(BLOCK) {
        #[allow(clippy::cast_possible_truncation)]
        let block_end = (block_start + BLOCK).min(len) as u32;
        for slot in 0..=keys.len() {
            let bucket = &buckets[slot];
            let cursor = &mut bucket_cursor[slot];
            while let Some(&(row, j)) = bucket.get(*cursor).filter(|(row, _)| *row < block_end) {
                *cursor += 1;
                let (row, j) = (row as usize, j as usize);
                let index = offsets[row] as usize + fill[row] as usize;
                fill[row] += 1;
                key_views[index] = rebase(residual_key_views[j], residual_key_base);
                if let (Some(views), Some(valid)) = (&residual_views, &residual_value_valid) {
                    value_views[index] = rebase(views[j], residual_value_base);
                    if !valid.value(j) {
                        value_valid.unset(index);
                    }
                }
            }
            let Some(rows) = present_rows.get(slot) else {
                continue;
            };
            let key_view = column_key_views[slot];
            let mut column = columns.get_mut(slot);
            let cursor = &mut column_cursor[slot];
            while let Some(&row) = rows.get(*cursor).filter(|&&row| row < block_end) {
                *cursor += 1;
                let row = row as usize;
                if repeats.value(row) {
                    continue;
                }
                let index = offsets[row] as usize + fill[row] as usize;
                fill[row] += 1;
                key_views[index] = key_view;
                if let Some(column) = column.as_mut() {
                    value_views[index] = column.view(row);
                }
            }
        }
    }

    let utf8 = |views: Vec<BinaryView>, buffers: Vec<ByteBuffer>, dtype: DType, validity| {
        let buffers: Arc<[ByteBuffer]> = buffers.into();
        // SAFETY: every view is copied from a valid source view with its buffer index rebased
        // onto the concatenated buffer list, or built over a key in the pool buffer.
        vortex_array::IntoArray::into_array(unsafe {
            VarBinViewArray::new_unchecked(Buffer::from(views), buffers, dtype, validity)
        })
    };
    let keys_array = utf8(
        key_views,
        key_sink.buffers,
        DType::Utf8(vortex_array::dtype::Nullability::NonNullable),
        Validity::NonNullable,
    );
    let values = with_values.then(|| {
        let dtype = flat.map_dtype.value_dtype();
        let validity = if dtype.is_nullable() {
            Validity::from_bit_buffer(value_valid.freeze(), dtype.nullability())
        } else {
            Validity::NonNullable
        };
        utf8(value_views, value_sink.buffers, dtype, validity)
    });
    Ok(Columnar {
        offsets,
        sizes,
        keys: keys_array,
        values,
    })
}
