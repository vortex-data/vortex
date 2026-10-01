// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use num_traits::WrappingAdd;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;

#[derive(Debug)]
pub(crate) struct FoRIsSortedKernel;

impl DynAggregateKernel for FoRIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<FoR>() else {
            return Ok(None);
        };
        if array.constant_reference().is_none() && !is_constant(array.references(), ctx)? {
            let Some(result) = is_sorted_per_chunk(array, options.strict, ctx)? else {
                return Ok(None);
            };
            return Ok(Some(IsSorted::make_partial(
                batch,
                result,
                options.strict,
                ctx,
            )?));
        }

        let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
        let unsigned_array = PrimitiveArray::from_buffer_handle(
            encoded.buffer_handle().clone(),
            encoded.ptype().to_unsigned(),
            encoded.validity()?,
        )
        .into_array();

        let result = if options.strict {
            is_strict_sorted(&unsigned_array, ctx)?
        } else {
            is_sorted(&unsigned_array, ctx)?
        };

        Ok(Some(IsSorted::make_partial(
            batch,
            result,
            options.strict,
            ctx,
        )?))
    }
}

/// Whether an array with different per-chunk references is sorted, or `None` if it has nulls.
///
/// Adds each chunk's reference while checking, without writing the decoded values.
fn is_sorted_per_chunk(
    array: ArrayView<'_, FoR>,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<bool>> {
    let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
    if !encoded.all_valid(ctx)? {
        return Ok(None);
    }
    let references = array.references().clone().execute::<PrimitiveArray>(ctx)?;
    // The first chunk may be partial when the array was sliced.
    let first_len = (FL_CHUNK_SIZE - usize::from(array.offset())).min(array.len());
    Ok(Some(match_each_integer_ptype!(array.ptype(), |T| {
        if strict {
            is_sorted_typed::<T>(&encoded, &references, first_len, |a, b| a < b)
        } else {
            is_sorted_typed::<T>(&encoded, &references, first_len, |a, b| a <= b)
        }
    })))
}

fn is_sorted_typed<T: NativePType + WrappingAdd>(
    encoded: &PrimitiveArray,
    references: &PrimitiveArray,
    first_len: usize,
    in_order: impl Fn(T, T) -> bool + Copy,
) -> bool {
    let (first, rest) = encoded.as_slice::<T>().split_at(first_len);
    let mut last: Option<T> = None;
    for (chunk, &reference) in iter::once(first)
        .chain(rest.chunks(FL_CHUNK_SIZE))
        .zip(references.as_slice::<T>())
    {
        let (Some(&chunk_first), Some(&chunk_last)) = (chunk.first(), chunk.last()) else {
            continue;
        };
        // Check the whole chunk without branching, so it vectorizes.
        let chunk_in_order = chunk.windows(2).fold(true, |acc, pair| {
            acc & in_order(
                pair[0].wrapping_add(&reference),
                pair[1].wrapping_add(&reference),
            )
        });
        let boundary_in_order =
            last.is_none_or(|last| in_order(last, chunk_first.wrapping_add(&reference)));
        if !chunk_in_order || !boundary_in_order {
            return false;
        }
        last = Some(chunk_last.wrapping_add(&reference));
    }
    true
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::FoRData;
    use crate::r#for::array::FoRArraySlotsExt;

    #[test]
    fn test_sorted() {
        let mut ctx = array_session().create_execution_ctx();

        let a = PrimitiveArray::new(buffer![-1, 0, i8::MAX], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, i8::MAX], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, 30, 127], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, -3, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, -3, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, -11, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, i8::MIN, -1], Validity::NonNullable);
        let b = FoRData::encode(a, &mut ctx).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );
    }
}
