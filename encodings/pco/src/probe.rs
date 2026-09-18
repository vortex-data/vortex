// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! PCO probes retain every decoded page and index the compacted, non-null value positions.

use std::ops::Range;

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use pco::wrapped::FileDecompressor;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Pco;
use crate::PcoArrayExt;
use crate::array::number_type_from_ptype;
use crate::array::vortex_err_from_pco;

const RANK_STRIDE: usize = 512;

/// State retained by repeated PCO probes. Construction performs no allocation.
///
/// The mask and rank index use unsliced logical rows; page boundaries use compacted value
/// positions. Every page decoded so far is retained, so the state grows with the pages touched
/// and never beyond the decompressed array.
#[derive(Default)]
pub struct PcoProbeState {
    validity: Option<Mask>,
    rank: Vec<usize>,
    pages: Vec<Page>,
    last_page: usize,
}

struct Page {
    values: Range<usize>,
    chunk: usize,
    decoded: Option<PrimitiveArray>,
}

pub(crate) fn scalar_at(
    array: ArrayView<'_, Pco>,
    index: usize,
    state: Option<&mut PcoProbeState>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let Some(state) = state else {
        let validity = array.unsliced_validity();
        return array
            ._slice(index, index + 1)
            .decompress(&validity, ctx)?
            .into_array()
            .execute_scalar(0, ctx);
    };
    let mask = match &mut state.validity {
        Some(mask) => mask,
        slot @ None => slot.insert(
            array
                .unsliced_validity()
                .execute_mask(array.unsliced_n_rows(), ctx)?,
        ),
    };
    let logical_index = array.slice_start() + index;

    let value_index = match mask {
        Mask::AllTrue(_) => logical_index,
        Mask::AllFalse(_) => unreachable!("probe dispatch checks validity"),
        Mask::Values(values) => {
            let bits = values.bit_buffer();
            if state.rank.is_empty() {
                state.rank.reserve(bits.len().div_ceil(RANK_STRIDE));
                let mut count = 0;
                for start in (0..bits.len()).step_by(RANK_STRIDE) {
                    state.rank.push(count);
                    count += bits.count_range(start, (start + RANK_STRIDE).min(bits.len()));
                }
            }
            let block = logical_index / RANK_STRIDE;
            state.rank[block] + bits.count_range(block * RANK_STRIDE, logical_index)
        }
    };

    if state.pages.is_empty() {
        let mut start = 0;
        for (chunk, metadata) in array.metadata.chunks.iter().enumerate() {
            for page in &metadata.pages {
                let end = start + page.n_values as usize;
                state.pages.push(Page {
                    values: start..end,
                    chunk,
                    decoded: None,
                });
                start = end;
            }
        }
    }
    let page_index = if state.pages[state.last_page].values.contains(&value_index) {
        state.last_page
    } else {
        state
            .pages
            .partition_point(|page| page.values.end <= value_index)
    };
    state.last_page = page_index;
    let page = state
        .pages
        .get_mut(page_index)
        .ok_or_else(|| vortex_err!("Missing PCO page for value {value_index}"))?;
    let values = match &mut page.decoded {
        Some(decoded) => decoded,
        slot @ None => {
            let decoded = match_number_enum!(
                number_type_from_ptype(array.dtype().as_ptype()),
                NumberType<T> => {
                    decode_page::<T>(array, page.chunk, page.values.len(), array.pages[page_index].as_slice(), ctx)?
                }
            );
            slot.insert(decoded)
        }
    };
    Ok(match_each_native_ptype!(values.ptype(), |T| {
        Scalar::primitive(
            values.as_slice::<T>()[value_index - page.values.start],
            array.dtype().nullability(),
        )
    }))
}

fn decode_page<T: Number + NativePType>(
    array: ArrayView<'_, Pco>,
    chunk: usize,
    n_values: usize,
    buffer: &[u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let (file, _) =
        FileDecompressor::new(array.metadata.header.as_slice()).map_err(vortex_err_from_pco)?;
    let (mut chunk, _) = file
        .chunk_decompressor::<T, _>(array.chunk_metas[chunk].as_ref())
        .map_err(vortex_err_from_pco)?;
    let mut decoder = chunk
        .page_decompressor(buffer, n_values)
        .map_err(vortex_err_from_pco)?;
    let mut values = BufferMut::<T>::zeroed_in(n_values, ctx.allocator().clone());
    decoder.read(&mut values).map_err(vortex_err_from_pco)?;
    Ok(PrimitiveArray::new(values.freeze(), Validity::NonNullable))
}

#[cfg(test)]
mod tests;
