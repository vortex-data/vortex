// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive take that decodes only pages containing requested non-null values.

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::Pco;
use crate::PcoArrayExt;
use crate::access::AccessState;
use crate::array::number_type_from_ptype;
use crate::decoder::PageDecoder;
use crate::decoder::PageLookup;
use crate::page_index::LogicalPageIndex;
use crate::page_index::PageIndex;

impl TakeExecute for Pco {
    fn take(
        array: ArrayView<'_, Pco>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let PageIndex::IndexedFast(index) = array.page_index()? else {
            return Ok(None);
        };
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;
        let mask = indices.validity()?.execute_mask(indices.len(), ctx)?;
        let extrema = min_max(indices.as_array(), ctx, NumericalAggregateOpts::default())?;
        if let Some(extrema) = extrema {
            match_each_integer_ptype!(indices.ptype(), |I| {
                validate_extrema::<I>(&extrema, array.len())?
            });
        }
        if mask.all_false() || array.pages.is_empty() {
            return Ok(Some(
                ConstantArray::new(Scalar::null(array.dtype().as_nullable()), indices.len())
                    .into_array(),
            ));
        }
        let requests = match_each_integer_ptype!(indices.ptype(), |I| {
            indices
                .as_slice::<I>()
                .iter()
                .zip(mask.iter())
                .enumerate()
                .filter_map(|(output, (&row, valid))| valid.then_some((output, row)))
                .map(|(output, row)| (array.slice_start() + row as usize, output))
                .collect::<Vec<_>>()
        });
        let nullable = array.dtype().nullability() | indices.dtype().nullability();
        let result = match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
            if nullable.is_nullable() {
                gather::<T, true>(array, index.as_ref(), requests, indices.len(), ctx)?
            } else {
                gather::<T, false>(array, index.as_ref(), requests, indices.len(), ctx)?
            }
        });
        Ok(Some(result.into_array()))
    }
}

fn validate_extrema<I: NativePType>(extrema: &MinMaxResult, source_len: usize) -> VortexResult<()>
where
    usize: TryFrom<I>,
{
    let min = extrema
        .min
        .as_primitive()
        .try_typed_value::<I>()?
        .ok_or_else(|| vortex_err!("Pco index minimum is null"))?;
    let max = extrema
        .max
        .as_primitive()
        .try_typed_value::<I>()?
        .ok_or_else(|| vortex_err!("Pco index maximum is null"))?;
    checked_index(min)?;
    let max = checked_index(max)?;
    vortex_ensure!(
        max < source_len,
        "Pco take index {max} exceeds array length {source_len}"
    );
    Ok(())
}

fn checked_index<I>(value: I) -> VortexResult<usize>
where
    usize: TryFrom<I>,
{
    usize::try_from(value).map_err(|_| vortex_err!("Pco take index is negative or exceeds usize"))
}

/// Sort requests once, resolve each repeated logical position once, then invoke the native
/// primitive gather once per physical page. Only the final scatter visits output positions;
/// it does no page-index search, validity rank, decoder lookup, or cache lookup.
fn gather<T: Number + NativePType, const NULLABLE: bool>(
    array: ArrayView<'_, Pco>,
    view: &LogicalPageIndex,
    mut requests: Vec<(usize, usize)>,
    output_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    requests.sort_unstable();
    let mut state = AccessState::default();
    let mut resolved = Vec::with_capacity(requests.len().min(64));
    let mut cursor = 0;
    while cursor < requests.len() {
        let row = requests[cursor].0;
        let end = cursor + requests[cursor..].partition_point(|request| request.0 == row);
        if let Some((page, value)) = state.resolve::<true>(view, array, row, ctx)? {
            resolved.push((page, value, cursor..end));
        }
        cursor = end;
    }
    let mut values = BufferMut::<T>::zeroed_in(output_len, ctx.allocator().clone());
    let mut valid = BitBufferMut::new_unset(if NULLABLE { output_len } else { 0 });
    // Requests are already in page order. Release all validity/rank preparation before
    // decoding, and keep only the current decoded page while scattering each group.
    drop(state);
    if !resolved.is_empty() {
        let mut decoder = PageDecoder::<T>::new(array.metadata.header.as_slice())?;
        let mut cursor = 0;
        let mut lookup = PageLookup::default();
        while cursor < resolved.len() {
            let page = resolved[cursor].0;
            let end = cursor + resolved[cursor..].partition_point(|request| request.0 == page);
            let compact = PrimitiveArray::from_iter(
                resolved[cursor..end]
                    .iter()
                    .map(|request| {
                        u64::try_from(request.1)
                            .map_err(|_| vortex_err!("Pco compact index exceeds u64"))
                    })
                    .collect::<VortexResult<Vec<_>>>()?,
            )
            .into_array();
            let (chunk_id, n_values) = lookup.resolve(view, array, page)?;
            let buffer = array
                .pages
                .get(page)
                .ok_or_else(|| vortex_err!("Pco compressed page is missing"))?;
            let chunk = array
                .chunk_metas
                .get(chunk_id)
                .ok_or_else(|| vortex_err!("Pco chunk metadata is missing"))?;
            let decoded = decoder.read(
                chunk_id,
                chunk.as_slice(),
                n_values,
                buffer.as_slice(),
                ctx,
            )?;
            let selected = decoded
                .into_array()
                .take(compact)?
                .execute::<PrimitiveArray>(ctx)?;
            for (request, &value) in resolved[cursor..end].iter().zip(selected.as_slice::<T>()) {
                for &(_, output) in &requests[request.2.clone()] {
                    values[output] = value;
                    if NULLABLE {
                        valid.set(output);
                    }
                }
            }
            cursor = end;
        }
    }
    let validity = if NULLABLE {
        Validity::from(valid.freeze())
    } else {
        Validity::NonNullable
    };
    Ok(PrimitiveArray::new(values.freeze(), validity))
}
