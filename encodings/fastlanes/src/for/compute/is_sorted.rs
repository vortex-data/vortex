// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::FoR;
use crate::r#for::array::FoRArrayExt;

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
    use crate::r#for::array::FoRArrayExt;

    #[test]
    fn test_sorted() {
        let mut ctx = array_session().create_execution_ctx();

        let a = PrimitiveArray::new(buffer![-1, 0, i8::MAX], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, i8::MAX], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, 0, 30, 127], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![i8::MIN, -3, -1], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, -3, -1], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, -11, -1], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );

        let a = PrimitiveArray::new(buffer![-10, i8::MIN, -1], Validity::NonNullable);
        let b = FoRData::encode(a).unwrap();
        assert!(
            !is_sorted(&b.clone().into_array(), &mut ctx).unwrap(),
            "{}",
            b.encoded().display_values()
        );
    }
}
