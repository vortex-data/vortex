// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate binding for extension dtypes defined across workspace crates.

use std::num::NonZeroUsize;
use std::sync::Arc;

use rstest::rstest;
use vortex_array::EmptyMetadata;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_error::VortexResult;
use vortex_json::Json;
use vortex_tensor::vector::Vector;
use vortex_uuid::Uuid;

#[rstest]
#[case::timestamp(DType::Extension(Timestamp::new(TimeUnit::Milliseconds, NonNullable).erased()), true, true)]
#[case::uuid(DType::Extension(ExtDType::<Uuid>::try_new(Default::default(), NonNullable)?.erased()), true, true)]
#[case::json(DType::Extension(ExtDType::<Json>::try_new(EmptyMetadata, NonNullable)?.erased()), false, false)]
#[case::vector(vector_dtype()?, false, true)]
fn aggregates_check_claims(
    #[case] dtype: DType,
    #[case] ordered: bool,
    #[case] equatable: bool,
) -> VortexResult<()> {
    let numerical = NumericalAggregateOpts::default();
    let max_bytes = NonZeroUsize::MIN;
    assert_binding(MinMax, numerical, &dtype, ordered);
    assert_binding(Min, numerical, &dtype, ordered);
    assert_binding(Max, numerical, &dtype, ordered);
    assert_binding(BoundedMin, BoundedMinOptions { max_bytes }, &dtype, ordered);
    assert_binding(BoundedMax, BoundedMaxOptions { max_bytes }, &dtype, ordered);
    assert_binding(IsSorted, IsSortedOptions { strict: false }, &dtype, ordered);
    assert_binding(IsConstant, EmptyOptions, &dtype, equatable);
    let pair = DType::struct_([("lhs", dtype.clone()), ("rhs", dtype)], NonNullable);
    assert_binding(AllNonDistinct, EmptyOptions, &pair, equatable);
    Ok(())
}

fn vector_dtype() -> VortexResult<DType> {
    let storage = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::F32, NonNullable)),
        3,
        NonNullable,
    );
    Ok(DType::Extension(
        ExtDType::<Vector>::try_from_storage((), storage)?.erased(),
    ))
}

#[track_caller]
fn assert_binding<A: AggregateFnVTable>(
    aggregate: A,
    options: A::Options,
    dtype: &DType,
    accepted: bool,
) {
    let id = aggregate.id();
    assert_eq!(
        Accumulator::try_new(aggregate, options, dtype.clone()).is_ok(),
        accepted,
        "{id}"
    );
}
