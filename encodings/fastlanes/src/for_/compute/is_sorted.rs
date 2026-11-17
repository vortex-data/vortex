// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted_with_options;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::FoR;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

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
        // TODO(mk): support many references.
        if !array.references().is_global() {
            return Ok(None);
        }

        let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
        let unsigned_array = PrimitiveArray::from_buffer_handle(
            encoded.buffer_handle().clone(),
            encoded.ptype().to_unsigned(),
            encoded.validity()?,
        )
        .into_array();

        let result = is_sorted_with_options(&unsigned_array, *options, ctx)?;

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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use vortex_array::aggregate_fn::fns::is_sorted::SortDirection;
    use vortex_array::aggregate_fn::fns::is_sorted::is_sorted_with_options;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FoRData;
    use crate::for_::array::FoRArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::asc_crosses_zero(buffer![-1i8, 0, i8::MAX], SortDirection::Asc, true)]
    #[case::asc_full_range(buffer![i8::MIN, 0, i8::MAX], SortDirection::Asc, true)]
    #[case::asc_four_values(buffer![i8::MIN, 0, 30, 127], SortDirection::Asc, true)]
    #[case::asc_negative_from_min(buffer![i8::MIN, -3, -1], SortDirection::Asc, true)]
    #[case::asc_negative(buffer![-10i8, -3, -1], SortDirection::Asc, true)]
    #[case::asc_unsorted(buffer![-10i8, -11, -1], SortDirection::Asc, false)]
    #[case::asc_unsorted_through_min(buffer![-10i8, i8::MIN, -1], SortDirection::Asc, false)]
    #[case::desc_full_range(buffer![i8::MAX, 0, -1, i8::MIN], SortDirection::Desc, true)]
    #[case::desc_negative(buffer![-1i8, -3, -10], SortDirection::Desc, true)]
    #[case::desc_ascending_input(buffer![i8::MIN, 0, i8::MAX], SortDirection::Desc, false)]
    fn test_sorted(
        #[case] values: Buffer<i8>,
        #[case] direction: SortDirection,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FoRData::encode(PrimitiveArray::new(values, Validity::NonNullable), &mut ctx)?;
        let options = IsSortedOptions::default().with_direction(direction);

        assert_eq!(
            is_sorted_with_options(&array.clone().into_array(), options, &mut ctx)?,
            expected,
            "{}",
            array.encoded().display_values(&ctx)
        );
        Ok(())
    }
}
