// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoding a shredded map back into a canonical map by merging its columns and residual.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::match_each_integer_ptype;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_buffer::BitBuffer;
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
use crate::rowcmp::RowCmp;
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

/// Marks the rows whose residual entries and column values all equal the previous row's.
pub(crate) fn repeated_rows(
    flat: &FlatMap,
    column_arrays: &[ArrayRef],
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    let mut repeats = BitBuffer::collect_bool(flat.len, |row| flat.repeats_prev(row));
    for column in column_arrays {
        if repeats.true_count() == 0 {
            break;
        }
        repeats = &repeats & &RowCmp::new(column, ctx)?.same_as_prev_bits(flat.len);
    }
    Ok(repeats)
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
    let total = (0..flat.len)
        .filter(|&row| !repeats.value(row))
        .map(|row| flat.range(row).len() + (0..keys.len()).filter(|&k| present(row, k)).count())
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
        for k in (0..keys.len()).filter(|&k| present(row, k)) {
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
    // Dictionary columns are gathered from their values through their codes, so only the
    // referenced rows' codes are read and the per-row values are never materialized.
    let mut codes: Vec<Option<Vec<u32>>> = Vec::with_capacity(columns.len());
    for (column, array) in columns.iter().zip(column_arrays) {
        let (array, column_codes) = match array.as_opt::<Dict>() {
            Some(dict) => (dict.values().clone(), Some(codes_u32(dict.codes(), ctx)?)),
            None => (array.clone(), None),
        };
        codes.push(column_codes);
        sources.push(match column.variant {
            Some(child) => Source::Variant { child, array },
            None => Source::Array(array),
        });
    }
    if codes.iter().all(Option::is_none) {
        return gather(value_dtype, sources, &plan.take, ctx);
    }
    let mut positions = Positions::with_capacity(plan.take.len());
    for (&src, &pos) in plan.take.src.iter().zip(&plan.take.pos) {
        let pos = match src.checked_sub(1).and_then(|k| codes[k as usize].as_ref()) {
            Some(codes) => codes[pos as usize] as usize,
            None => pos as usize,
        };
        positions.push(src, pos);
    }
    gather(value_dtype, sources, &positions, ctx)
}

/// The codes of a dictionary as `u32`, with nulls read as zero.
pub(crate) fn codes_u32(codes: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u32>> {
    let codes = codes.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(codes.ptype(), |P| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    let repeats = repeated_rows(&flat, &parts.column_arrays, ctx)?;
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
    let repeats = repeated_rows(&flat, &parts.column_arrays, ctx)?;
    let plan = plan_merge(&flat, &keys, &masks, &repeats, false);
    build_listview(
        utf8_from_views(plan.key_views, plan.key_buffers),
        plan.offsets,
        plan.sizes,
        flat.validity,
    )
}
