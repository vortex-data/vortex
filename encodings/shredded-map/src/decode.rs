// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding a shredded map back into a canonical map by merging its columns and residual.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ShreddedColumn;
use crate::ShreddedMap;
use crate::array::ShreddedMapArraySlotsExt;
use crate::flat::FlatMap;
use crate::flat::Strings;
use crate::flat::build_listview;
use crate::flat::build_map;
use crate::gather::Positions;
use crate::gather::Source;
use crate::gather::gather;
use crate::flat::utf8_from_views;

/// The borrowed parts of a shredded map.
pub(crate) struct ShreddedParts<'a> {
    pub map_dtype: MapDType,
    pub columns: &'a [ShreddedColumn],
    pub column_arrays: Vec<ArrayRef>,
    pub residual: ArrayRef,
}

impl<'a> ShreddedParts<'a> {
    pub fn from_view(array: ArrayView<'a, ShreddedMap>) -> Self {
        let map_dtype = array
            .dtype()
            .as_map_opt()
            .cloned()
            .unwrap_or_else(|| unreachable!("ShreddedMap requires a map dtype"));
        Self {
            map_dtype,
            columns: array.data().columns(),
            column_arrays: array.columns().to_vec(),
            residual: array.residual().clone(),
        }
    }
}

/// Row-major merge of shredded columns and residual entries.
pub(crate) struct MergePlan {
    pub offsets: Vec<u64>,
    pub sizes: Vec<u64>,
    /// Views of the output keys, pointing into the residual key buffers plus one key pool buffer.
    pub key_views: Vec<BinaryView>,
    pub key_buffers: Arc<[ByteBuffer]>,
    /// For each output entry, its source (0 for the residual, `k + 1` for column `k`) and its
    /// position there: the residual entry index, or the row for a column. Empty when not
    /// requested.
    pub take: Positions,
}

pub(crate) fn column_masks(
    column_arrays: &[ArrayRef],
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Mask>> {
    column_arrays
        .iter()
        .map(|c| c.validity()?.execute_mask(len, ctx))
        .collect()
}

fn for_each_set(mask: &Mask, mut f: impl FnMut(usize)) {
    match mask.bit_buffer() {
        AllOr::All => (0..mask.len()).for_each(f),
        AllOr::None => {}
        AllOr::Some(bits) => bits.for_each_set_index(&mut f),
    }
}

/// Transposes the column masks into per-row lists of present column indices, in column order.
fn present_columns(masks: &[Mask], len: usize) -> (Vec<u32>, Vec<u16>) {
    let mut starts = vec![0u32; len + 1];
    for mask in masks {
        for_each_set(mask, |row| starts[row + 1] += 1);
    }
    for row in 0..len {
        starts[row + 1] += starts[row];
    }
    let mut cursor = starts.clone();
    let mut columns = vec![0u16; starts[len] as usize];
    for (k, mask) in masks.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let k = k as u16;
        for_each_set(mask, |row| {
            columns[cursor[row] as usize] = k;
            cursor[row] += 1;
        });
    }
    (starts, columns)
}

pub(crate) fn plan_merge(
    flat: &FlatMap,
    keys: &[&str],
    column_masks: &[Mask],
    with_take: bool,
) -> MergePlan {
    let residual_keys = Strings::new(&flat.keys);
    let pool_index = u32::try_from(residual_keys.buffers.len()).unwrap_or(u32::MAX);
    let mut pool = Vec::new();
    let column_views: Vec<BinaryView> = keys
        .iter()
        .map(|key| {
            let offset = u32::try_from(pool.len()).unwrap_or(u32::MAX);
            pool.extend_from_slice(key.as_bytes());
            BinaryView::make_view(key.as_bytes(), pool_index, offset)
        })
        .collect();

    let total = (0..flat.len)
        .map(|row| flat.range(row).len())
        .sum::<usize>()
        + column_masks.iter().map(Mask::true_count).sum::<usize>();
    let mut offsets = Vec::with_capacity(flat.len);
    let mut sizes = Vec::with_capacity(flat.len);
    let mut key_views = Vec::with_capacity(total);
    let mut take = Positions::with_capacity(if with_take { total } else { 0 });

    let (row_starts, row_columns) = present_columns(column_masks, flat.len);
    for row in 0..flat.len {
        let start = key_views.len();
        let range = flat.range(row);
        let mut j = range.start;
        for &k in &row_columns[row_starts[row] as usize..row_starts[row + 1] as usize] {
            let k = k as usize;
            let column_key = keys[k].as_bytes();
            while j < range.end && residual_keys.get(j) < column_key {
                key_views.push(residual_keys.views[j]);
                if with_take {
                    take.push(0, j);
                }
                j += 1;
            }
            key_views.push(column_views[k]);
            if with_take {
                #[allow(clippy::cast_possible_truncation)]
                take.push(k as u16 + 1, row);
            }
        }
        for j in j..range.end {
            key_views.push(residual_keys.views[j]);
            if with_take {
                take.push(0, j);
            }
        }
        offsets.push(start as u64);
        sizes.push((key_views.len() - start) as u64);
    }

    let mut key_buffers: Vec<ByteBuffer> = flat
        .keys
        .data_buffers()
        .iter()
        .map(|b| b.as_host().clone())
        .collect();
    key_buffers.push(ByteBuffer::from(pool));

    MergePlan {
        offsets,
        sizes,
        key_views,
        key_buffers: key_buffers.into(),
        take,
    }
}

/// Gathers the output values of a merge plan.
fn gather_values(
    value_dtype: &DType,
    flat: &FlatMap,
    columns: &[ShreddedColumn],
    column_arrays: &[ArrayRef],
    plan: &MergePlan,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut sources = Vec::with_capacity(columns.len() + 1);
    sources.push(Source::Array(flat.values.clone()));
    for (column, array) in columns.iter().zip(column_arrays) {
        sources.push(match column.variant {
            Some(child) => Source::Variant {
                child,
                array: array.clone(),
            },
            None => Source::Array(array.clone()),
        });
    }
    gather(value_dtype, sources, &plan.take, ctx)
}

/// Decodes shredded parts into a canonical map array.
pub(crate) fn decode_parts(
    parts: &ShreddedParts<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<MapArray> {
    if parts.columns.is_empty() {
        return parts.residual.clone().execute::<MapArray>(ctx);
    }
    let flat = FlatMap::new(&parts.residual, ctx)?;
    let len = flat.len;
    let keys: Vec<&str> = parts.columns.iter().map(|c| c.key.as_ref()).collect();
    let masks = column_masks(&parts.column_arrays, len, ctx)?;
    let plan = plan_merge(&flat, &keys, &masks, true);
    let values = gather_values(
        &parts.map_dtype.value_dtype(),
        &flat,
        parts.columns,
        &parts.column_arrays,
        &plan,
        ctx,
    )?;
    let MergePlan {
        offsets,
        sizes,
        key_views,
        key_buffers,
        ..
    } = plan;
    build_map(
        &parts.map_dtype,
        utf8_from_views(key_views, key_buffers),
        values,
        offsets,
        sizes,
        flat.validity.clone(),
    )
}

pub(crate) fn decode_to_map(
    array: ArrayView<'_, ShreddedMap>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<MapArray> {
    decode_parts(&ShreddedParts::from_view(array), ctx)
}

/// Decodes only the keys of each row, as a `List<Utf8>`.
pub(crate) fn decode_keys(
    parts: &ShreddedParts<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    let flat = FlatMap::new(&parts.residual, ctx)?;
    let keys: Vec<&str> = parts.columns.iter().map(|c| c.key.as_ref()).collect();
    let masks = column_masks(&parts.column_arrays, flat.len, ctx)?;
    let plan = plan_merge(&flat, &keys, &masks, false);
    build_listview(
        utf8_from_views(plan.key_views, plan.key_buffers),
        plan.offsets,
        plan.sizes,
        flat.validity,
    )
}
