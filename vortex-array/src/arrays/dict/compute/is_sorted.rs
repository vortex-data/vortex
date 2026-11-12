// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::is_sorted::SortDirection;
use crate::aggregate_fn::fns::is_sorted::is_sorted_with_options;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar::Scalar;

/// Dict-specific is_sorted kernel.
///
/// | Values     | Codes     | Dict array sortedness |
/// |------------|-----------|-----------------------|
/// | Ascending  | Ascending | Ascending             |
/// | Descending | Ascending | Descending            |
///
/// Values and codes must be sorted with the same strictness.
#[derive(Debug)]
pub(crate) struct DictIsSortedKernel;

impl DynAggregateKernel for DictIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let strict = options.strict;

        // Detect early false. Strict order allows at most one null.
        if strict && batch.invalid_count(ctx)? > 1 {
            return Ok(Some(IsSorted::make_partial(batch, false, strict, ctx)?));
        }

        let sorted = match options.direction {
            SortDirection::Asc => {
                is_sorted_with_options(dict.values(), *options, ctx)?
                    && is_sorted_with_options(dict.codes(), *options, ctx)?
            }
            SortDirection::Desc => {
                // `DictEncoder` produce incremental asc codes, so a desc
                // column gets `DESC` values and `ASC` codes. So, using `DESC` codes,
                // e.g. from `take` or an imported dict, are out of scope.
                let asc_options = IsSortedOptions {
                    strict,
                    direction: SortDirection::Asc,
                };
                is_sorted_with_options(dict.values(), *options, ctx)?
                    && is_sorted_with_options(dict.codes(), asc_options, ctx)?
            }
        };

        if sorted {
            Ok(Some(IsSorted::make_partial(batch, true, strict, ctx)?))
        } else {
            // We can't definitively say it's NOT sorted without canonicalizing,
            // so return None to let the accumulator handle it.
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DictIsSortedKernel;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::is_sorted::SortDirection;
    use crate::aggregate_fn::fns::is_sorted::is_sorted_with_options;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::ascending(buffer![1i32, 5, 9], buffer![0u32, 1, 2], SortDirection::Asc, Some(true))]
    #[case::descending(buffer![9i32, 5, 1], buffer![0u32, 1, 2], SortDirection::Desc, Some(true))]
    fn kernel_agrees_with_decoded(
        #[case] values: Buffer<i32>,
        #[case] codes: Buffer<u32>,
        #[case] direction: SortDirection,
        #[case] verdict: Option<bool>,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let options = IsSortedOptions { strict, direction };
        let dict = DictArray::try_new(codes.into_array(), values.into_array())?.into_array();

        let decoded = dict.clone().execute::<Canonical>(&mut ctx)?.into_array();
        let sorted = is_sorted_with_options(&decoded, options, &mut ctx)?;
        assert!(verdict.is_none_or(|verdict| verdict == sorted));

        let expected = verdict
            .map(|verdict| IsSorted::make_partial(&dict, verdict, strict, &mut ctx))
            .transpose()?;
        let result = DictIsSortedKernel.aggregate(&IsSorted.bind(options), &dict, &mut ctx)?;
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn strict_repeated_nulls_are_not_sorted() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(1)]).into_array(),
            PrimitiveArray::from_option_iter([None, Some(5i32)]).into_array(),
        )?
        .into_array();
        let options = IsSortedOptions {
            strict: true,
            ..Default::default()
        };

        let result = DictIsSortedKernel.aggregate(&IsSorted.bind(options), &dict, &mut ctx)?;
        assert_eq!(
            result,
            Some(IsSorted::make_partial(&dict, false, true, &mut ctx)?)
        );
        Ok(())
    }
}
