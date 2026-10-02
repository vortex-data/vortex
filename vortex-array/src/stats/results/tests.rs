// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::AggregateResults;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;

#[test]
fn options_are_part_of_the_key() -> VortexResult<()> {
    let skip = Sum.bind(NumericalAggregateOpts::skip_nans());
    let include = Sum.bind(NumericalAggregateOpts::include_nans());
    let value = Scalar::primitive(3.0f64, Nullability::Nullable);
    let results = AggregateResults::try_new(
        &DType::from(PType::F64),
        [(skip.clone(), Precision::Exact(value.clone()))],
    )?;
    assert_eq!(results.get_result(&skip), Precision::Exact(value));
    assert_eq!(results.get_result(&include), Precision::Absent);
    Ok(())
}

#[test]
fn missing_differs_from_an_exact_null() -> VortexResult<()> {
    let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
    let value = Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable));
    let results = AggregateResults::try_new(
        &DType::from(PType::I32),
        [
            (sum.clone(), Precision::Exact(value.clone())),
            (NullCount.bind(EmptyOptions), Precision::Absent),
        ],
    )?;
    assert_eq!(results.iter().count(), 1);
    assert_eq!(results.get_result(&sum), Precision::Exact(value));
    assert_eq!(
        results.get_result(&NullCount.bind(EmptyOptions)),
        Precision::Absent
    );
    Ok(())
}

#[test]
fn duplicate_and_incompatible_results_are_rejected() {
    let dtype = DType::from(PType::I32);
    let count = NullCount.bind(EmptyOptions);
    let entry = (count.clone(), Precision::Exact(1u64.into()));
    assert!(AggregateResults::try_new(&dtype, [entry.clone(), entry]).is_err());
    assert!(AggregateResults::try_new(&dtype, [(count, Precision::Exact(true.into()))]).is_err());
}

#[test]
fn sortedness_is_a_final_result() -> VortexResult<()> {
    let dtype = DType::from(PType::I32);
    let sorted = IsSorted.bind(IsSortedOptions { strict: false });
    let results =
        AggregateResults::try_new(&dtype, [(sorted.clone(), Precision::Exact(true.into()))])?;
    let value = results.get_result(&sorted).as_exact().unwrap();
    assert_eq!(value.dtype(), &DType::Bool(Nullability::NonNullable));
    assert_ne!(sorted.state_dtype(&dtype), sorted.return_dtype(&dtype));
    Ok(())
}
