// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding a shredded map back into a canonical map by merging its columns and residual.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;

use crate::ShreddedColumn;
use crate::ShreddedMap;
use crate::array::ShreddedMapArrayExt;
use crate::array::ShreddedMapArraySlotsExt;
use crate::columnar::decode_columnar;
use crate::flat::FlatMap;
use crate::flat::Strings;
use crate::flat::build_listview;
use crate::flat::build_map;
use crate::flat::to_usize_vec;
use crate::flat::utf8_from_views;
use crate::gather::Positions;
use crate::gather::Source;
use crate::gather::gather;
use crate::rowcmp::RowCmp;

/// The borrowed parts of a shredded map.
pub(crate) struct ShreddedParts<'a> {
    pub map_dtype: MapDType,
    pub columns: &'a [ShreddedColumn],
    pub column_arrays: Vec<ArrayRef>,
    pub residual: ArrayRef,
    pub repeats: Option<ArrayRef>,
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
            column_arrays: array.column_arrays(),
            residual: array.residual().clone(),
            repeats: array.repeats().cloned(),
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

/// Marks the rows whose residual entries and column values all equal the previous row's.
pub(crate) fn repeated_rows(
    flat: &FlatMap,
    column_arrays: &[ArrayRef],
    hint: Option<&ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    // The shredder records which rows it made share entries; trust that instead of re-deriving it.
    if let Some(hint) = hint {
        let bits = hint.clone().execute::<BoolArray>(ctx)?.to_bit_buffer();
        // Row 0 never repeats, whatever a slice left there.
        return Ok(BitBuffer::collect_bool(flat.len, |row| {
            row > 0 && bits.value(row)
        }));
    }
    let comparators = column_arrays
        .iter()
        .map(|c| RowCmp::new(c, ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    // Sharing rows only pays off when most rows repeat, so estimate that on a sample first.
    let step = (flat.len / 4096).max(1);
    let sample: Vec<usize> = (1..flat.len).step_by(step).collect();
    let hits = sample
        .iter()
        .filter(|&&row| flat.repeats_prev(row) && comparators.iter().all(|c| c.equal(row, row - 1)))
        .count();
    if hits * 2 < sample.len() {
        return Ok(BitBuffer::new_unset(flat.len));
    }
    let mut repeats = BitBuffer::collect_bool(flat.len, |row| flat.repeats_prev(row));
    for comparator in &comparators {
        repeats = &repeats & &comparator.same_as_prev_bits(flat.len);
    }
    Ok(repeats)
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
    repeats: &BitBuffer,
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

    // Only rows that do not repeat their predecessor are merged, so their present columns are
    // looked up per row instead of transposing every column mask.
    let column_bits: Vec<AllOr<&BitBuffer>> = column_masks.iter().map(Mask::bit_buffer).collect();
    let present = |row: usize, k: usize| match column_bits[k] {
        AllOr::All => true,
        AllOr::None => false,
        AllOr::Some(bits) => bits.value(row),
    };
    // Per-row column lists cost one pass over the present entries; per-row lookups cost one check
    // per column for every row that does not repeat. Build the lists when they are cheaper.
    let merged_rows = flat.len - repeats.true_count();
    let present_entries: usize = column_masks.iter().map(Mask::true_count).sum();
    let row_columns = (merged_rows.saturating_mul(keys.len()) > present_entries)
        .then(|| present_columns(column_masks, flat.len));
    let mut row_buf: Vec<usize> = Vec::with_capacity(keys.len());
    let columns_of = |row: usize, buf: &mut Vec<usize>| {
        buf.clear();
        match &row_columns {
            Some((starts, cols)) => buf.extend(
                cols[starts[row] as usize..starts[row + 1] as usize]
                    .iter()
                    .map(|&k| k as usize),
            ),
            None => buf.extend((0..keys.len()).filter(|&k| present(row, k))),
        }
    };
    let total = (0..flat.len)
        .filter(|&row| !repeats.value(row))
        .map(|row| {
            columns_of(row, &mut row_buf);
            flat.range(row).len() + row_buf.len()
        })
        .sum::<usize>();
    let mut offsets = Vec::with_capacity(flat.len);
    let mut sizes = Vec::with_capacity(flat.len);
    let mut key_views = Vec::with_capacity(total);
    let mut take = Positions::with_capacity(if with_take { total } else { 0 });

    for row in 0..flat.len {
        if repeats.value(row) {
            offsets.push(offsets[row - 1]);
            sizes.push(sizes[row - 1]);
            continue;
        }
        let start = key_views.len();
        let range = flat.range(row);
        let mut j = range.start;
        columns_of(row, &mut row_buf);
        for &k in &row_buf {
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
    // Sparse and dictionary columns are gathered from their innermost values: a sparse layer maps
    // a row to its rank among the present rows, a dictionary layer maps a position to its code.
    // Only the referenced positions are read and per-row values are never materialized.
    let mut remaps: Vec<Vec<Remap>> = Vec::with_capacity(columns.len());
    for (column, array) in columns.iter().zip(column_arrays) {
        let (array, layers) = unwrap_layers(array, ctx)?;
        remaps.push(layers);
        sources.push(match column.variant {
            Some(child) => Source::Variant { child, array },
            None => Source::Array(array),
        });
    }
    if remaps.iter().all(Vec::is_empty) {
        return gather(value_dtype, sources, &plan.take, ctx);
    }
    // Column positions arrive in ascending row order, so a sparse outermost layer is resolved
    // with a moving cursor instead of a search.
    let mut cursors = vec![0usize; remaps.len()];
    let mut positions = Positions::with_capacity(plan.take.len());
    for (&src, &pos) in plan.take.src.iter().zip(&plan.take.pos) {
        let mut pos = pos as usize;
        if let Some(k) = src.checked_sub(1) {
            let k = k as usize;
            let mut layers = remaps[k].iter();
            if let Some(Remap::Rank(rows)) = remaps[k].first() {
                let cursor = &mut cursors[k];
                if rows.get(*cursor).is_some_and(|&r| r > pos) {
                    *cursor = rows.partition_point(|&r| r < pos);
                }
                while rows.get(*cursor).is_some_and(|&r| r < pos) {
                    *cursor += 1;
                }
                pos = *cursor;
                layers.next();
            }
            for remap in layers {
                pos = remap.apply(pos);
            }
        }
        positions.push(src, pos);
    }
    gather(value_dtype, sources, &positions, ctx)
}

/// A position mapping through one encoding layer of a column.
pub(crate) enum Remap {
    /// A dictionary: position to code.
    Codes(Vec<u32>),
    /// A sparse array: row to its rank among the ascending present rows.
    Rank(Vec<usize>),
}

impl Remap {
    #[inline]
    pub fn apply(&self, pos: usize) -> usize {
        match self {
            Self::Codes(codes) => codes[pos] as usize,
            Self::Rank(rows) => rows.partition_point(|&r| r < pos),
        }
    }
}

/// Peels sparse and dictionary layers off a column, returning its innermost values and the
/// position mappings to reach them, outermost first.
pub(crate) fn unwrap_layers(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Vec<Remap>)> {
    let mut array = array.clone();
    let mut layers = Vec::new();
    loop {
        if let Some(dict) = array.as_opt::<Dict>() {
            layers.push(Remap::Codes(codes_u32(dict.codes(), ctx)?));
            array = dict.values().clone();
            continue;
        }
        if let Some(sparse) = array.as_opt::<Sparse>()
            && sparse.patches().offset() == 0
            && sparse.fill_scalar().is_null()
        {
            let patches = sparse.patches();
            let rows = to_usize_vec(patches.indices(), ctx)?;
            let values = patches.values().clone();
            layers.push(Remap::Rank(rows));
            array = values;
            continue;
        }
        return Ok((array, layers));
    }
}

/// The codes of a dictionary as `u32`, with nulls read as zero.
pub(crate) fn codes_u32(codes: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u32>> {
    let codes = codes.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(codes.ptype(), |P| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::unnecessary_cast
        )]
        codes.as_slice::<P>().iter().map(|&c| c as u32).collect()
    }))
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
    let repeats = repeated_rows(&flat, &parts.column_arrays, parts.repeats.as_ref(), ctx)?;
    // Column-major decoding touches every present entry, while the row-major merge skips
    // repeated rows outright, so it wins when most rows repeat.
    let mostly_repeats = repeats.true_count() * 2 > len;
    if !mostly_repeats && matches!(parts.map_dtype.value_dtype(), DType::Utf8(_)) {
        let c = decode_columnar(
            &flat,
            &keys,
            &parts.column_arrays,
            &masks,
            &repeats,
            true,
            ctx,
        )?;
        return build_map(
            &parts.map_dtype,
            c.keys,
            c.values.unwrap_or_else(|| unreachable!()),
            c.offsets,
            c.sizes,
            flat.validity.clone(),
        );
    }
    let plan = plan_merge(&flat, &keys, &masks, &repeats, true);
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
    let repeats = repeated_rows(&flat, &parts.column_arrays, parts.repeats.as_ref(), ctx)?;
    if repeats.true_count() * 2 > flat.len {
        let plan = plan_merge(&flat, &keys, &masks, &repeats, false);
        return build_listview(
            utf8_from_views(plan.key_views, plan.key_buffers),
            plan.offsets,
            plan.sizes,
            flat.validity,
        );
    }
    let c = decode_columnar(
        &flat,
        &keys,
        &parts.column_arrays,
        &masks,
        &repeats,
        false,
        ctx,
    )?;
    build_listview(c.keys, c.offsets, c.sizes, flat.validity)
}
