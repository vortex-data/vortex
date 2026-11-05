// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryRunEndArray;
pub use array::*;
pub use iter::trimmed_ends_iter;

mod array;
pub mod compress;
mod compute;
pub mod decompress_bool;
mod iter;
mod kernel;
pub mod ops;
mod probe;
mod rules;
#[cfg(test)]
#[cfg(not(codspeed))]
mod trace_tests;

#[doc(hidden)]
pub mod _benchmarking {
    pub use compute::filter::filter_run_end_primitive;
    pub use compute::take::take_indices_unchecked;

    use super::*;
}

use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize run-end encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(RunEnd);
    kernel::initialize(session);

    // Register the RunEnd-specific aggregate kernels.
    session.aggregate_fns().register_aggregate_kernel(
        RunEnd.id(),
        Some(MinMax.id()),
        &compute::min_max::RunEndMinMaxKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        RunEnd.id(),
        Some(IsConstant.id()),
        &compute::is_constant::RunEndIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        RunEnd.id(),
        Some(IsSorted.id()),
        &compute::is_sorted::RunEndIsSortedKernel,
    );
    for sum in [Sum.id(), SumV2.id()] {
        session.aggregate_fns().register_aggregate_kernel(
            RunEnd.id(),
            Some(sum),
            &compute::sum::RunEndSumKernel,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::fns::string::StringFn;
    use vortex_array::test_harness::check_metadata;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEnd;
    use crate::RunEndMetadata;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn unicode_lower_preserves_runs() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = DictArray::try_new(
            PrimitiveArray::from_iter([0u8, 1, 2, 0]).into_array(),
            VarBinViewArray::from_iter_nullable_str([Some("ΟΣ"), None, Some("İȾ")]).into_array(),
        )?
        .into_array();
        let input = RunEnd::try_new_offset_length(
            PrimitiveArray::from_iter([3u32, 5, 7, 10]).into_array(),
            values,
            1,
            8,
            &mut ctx,
        )?
        .into_array();
        let expression = StringFn::Lower.try_new_expr(EmptyOptions, [root()])?;
        let result = input
            .apply(&expression)?
            .execute_until::<RunEnd>(&mut ctx)?;
        assert!(result.is::<RunEnd>());
        assert_arrays_eq!(
            result,
            VarBinViewArray::from_iter_nullable_str([
                Some("ος"),
                Some("ος"),
                None,
                None,
                Some("i\u{307}ⱦ"),
                Some("i\u{307}ⱦ"),
                Some("ος"),
                Some("ος"),
            ]),
            &mut ctx
        );
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_runend_metadata() {
        check_metadata(
            "runend.metadata",
            &RunEndMetadata {
                ends_ptype: PType::U64 as i32,
                num_runs: u64::MAX,
                offset: u64::MAX,
            }
            .encode_to_vec(),
        );
    }
}
