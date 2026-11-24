// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Grouped accumulation of the aggregates in this crate.

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::DynGroupedAccumulator;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::GroupedAccumulator;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::combined::PairOptions;
use crate::count::Count;
use crate::mean::Mean;
use crate::null_count::NullCount;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn test_accumulate_list_with_null_group_non_nullable_partials() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Groups: [[1.0, null, 3.0], null, [4.0]]
    let elements =
        PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(3.0), Some(4.0)]).into_array();
    let groups = ListViewArray::try_new(
        elements,
        buffer![0u32, 3, 3].into_array(),
        buffer![3u32, 0, 1].into_array(),
        Validity::from_iter([true, false, true]),
        &mut ctx,
    )?
    .into_array();
    let elem_dtype = DType::Primitive(PType::F64, Nullable);

    let mut count_acc =
        GroupedAccumulator::try_new(Count, NumericalAggregateOpts::default(), elem_dtype.clone())?;
    count_acc.accumulate_list(&groups, &mut ctx)?;
    assert_arrays_eq!(
        count_acc.finish()?,
        PrimitiveArray::new(buffer![2u64, 0, 1], Validity::NonNullable),
        &mut ctx
    );

    let mut null_count_acc =
        GroupedAccumulator::try_new(NullCount, EmptyOptions, elem_dtype.clone())?;
    null_count_acc.accumulate_list(&groups, &mut ctx)?;
    assert_arrays_eq!(
        null_count_acc.finish()?,
        PrimitiveArray::new(buffer![1u64, 0, 0], Validity::NonNullable),
        &mut ctx
    );

    let mut mean_acc = GroupedAccumulator::try_new(
        Mean::combined(),
        PairOptions(
            NumericalAggregateOpts::default(),
            NumericalAggregateOpts::default(),
        ),
        elem_dtype,
    )?;
    mean_acc.accumulate_list(&groups, &mut ctx)?;
    assert_arrays_eq!(
        mean_acc.finish()?,
        PrimitiveArray::from_option_iter([Some(2.0f64), None, Some(4.0)]),
        &mut ctx
    );
    Ok(())
}
