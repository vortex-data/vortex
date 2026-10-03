// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray as _;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

#[derive(Clone, Copy, Debug)]
enum Accumulation {
    Direct,
    TypedMerge,
    ScalarMerge,
}

fn accumulate<V: AggregateFnVTable>(
    vtable: V,
    options: V::Options,
    batches: &[ArrayRef],
    accumulation: Accumulation,
) -> VortexResult<Accumulator<V>> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = batches[0].dtype().clone();
    let mut result = Accumulator::try_new(vtable.clone(), options.clone(), dtype.clone())?;

    for batch in batches {
        match accumulation {
            Accumulation::Direct => result.accumulate(batch, &mut ctx)?,
            Accumulation::TypedMerge | Accumulation::ScalarMerge => {
                let mut local =
                    Accumulator::try_new(vtable.clone(), options.clone(), dtype.clone())?;
                local.accumulate(batch, &mut ctx)?;
                if matches!(accumulation, Accumulation::TypedMerge) {
                    result.merge_from(&mut local)?;
                } else {
                    result.combine_partials(local.flush()?)?;
                }
            }
        }
    }

    Ok(result)
}

#[rstest]
#[case::finite(Some(99.0))]
#[case::null(None)]
#[case::nan(Some(f64::NAN))]
fn empty_constant_results(
    #[case] value: Option<f64>,
    #[values(true, false)] skip_nans: bool,
) -> VortexResult<()> {
    let scalar = Scalar::from(value);
    let batches = || [ConstantArray::new(scalar.clone(), 0).into_array()];
    let options = NumericalAggregateOpts { skip_nans };

    let mut min = accumulate(Min, options, &batches(), Accumulation::Direct)?;
    assert!(min.partial_scalar()?.is_null());
    assert!(min.finish()?.is_null());

    let mut max = accumulate(Max, options, &batches(), Accumulation::Direct)?;
    assert!(max.partial_scalar()?.is_null());
    assert!(max.finish()?.is_null());

    let mut extrema = accumulate(MinMax, options, &batches(), Accumulation::Direct)?;
    assert!(extrema.partial_scalar()?.is_null());
    assert!(extrema.finish()?.is_null());

    let mut constant = accumulate(IsConstant, EmptyOptions, &batches(), Accumulation::Direct)?;
    assert!(constant.partial_scalar()?.is_null());
    assert_eq!(
        constant.finish()?,
        Scalar::bool(false, Nullability::NonNullable)
    );

    for strict in [false, true] {
        let mut sorted = accumulate(
            IsSorted,
            IsSortedOptions { strict },
            &batches(),
            Accumulation::Direct,
        )?;
        assert!(sorted.partial_scalar()?.is_null());
        assert_eq!(
            sorted.finish()?,
            Scalar::bool(true, Nullability::NonNullable)
        );
    }

    Ok(())
}

#[rstest]
#[case::finite(Some(99.0), true)]
#[case::null(None, true)]
#[case::skip_nan(Some(f64::NAN), true)]
#[case::include_nan(Some(f64::NAN), false)]
fn empty_constant_preserves_extrema(
    #[case] value: Option<f64>,
    #[case] skip_nans: bool,
    #[values(0, 1, 2)] position: usize,
    #[values(
        Accumulation::Direct,
        Accumulation::TypedMerge,
        Accumulation::ScalarMerge
    )]
    accumulation: Accumulation,
) -> VortexResult<()> {
    let mut batches = vec![
        PrimitiveArray::from_option_iter([Some(1.0f64)]).into_array(),
        PrimitiveArray::from_option_iter([Some(2.0f64)]).into_array(),
    ];
    batches.insert(
        position,
        ConstantArray::new(Scalar::from(value), 0).into_array(),
    );

    let mut accumulator = accumulate(
        MinMax,
        NumericalAggregateOpts { skip_nans },
        &batches,
        accumulation,
    )?;
    assert_eq!(
        MinMaxResult::from_scalar(accumulator.finish()?)?,
        Some(MinMaxResult {
            min: Scalar::primitive(1.0f64, Nullability::NonNullable),
            max: Scalar::primitive(2.0f64, Nullability::NonNullable),
        })
    );

    Ok(())
}

#[rstest]
#[case::finite(Some(99.0))]
#[case::null(None)]
#[case::nan(Some(f64::NAN))]
fn empty_constant_preserves_flags(
    #[case] value: Option<f64>,
    #[values(0, 1, 2)] position: usize,
    #[values(
        Accumulation::Direct,
        Accumulation::TypedMerge,
        Accumulation::ScalarMerge
    )]
    accumulation: Accumulation,
) -> VortexResult<()> {
    let first = PrimitiveArray::from_option_iter([Some(1.0f64)]).into_array();
    let mut batches = vec![first.clone(), first];
    batches.insert(
        position,
        ConstantArray::new(Scalar::from(value), 0).into_array(),
    );

    let mut constant = accumulate(IsConstant, EmptyOptions, &batches, accumulation)?;
    assert_eq!(
        constant.finish()?,
        Scalar::bool(true, Nullability::NonNullable)
    );

    let mut batches = vec![
        PrimitiveArray::from_option_iter([Some(1.0f64)]).into_array(),
        PrimitiveArray::from_option_iter([Some(2.0f64)]).into_array(),
    ];
    batches.insert(
        position,
        ConstantArray::new(Scalar::from(value), 0).into_array(),
    );
    for strict in [false, true] {
        let mut sorted = accumulate(IsSorted, IsSortedOptions { strict }, &batches, accumulation)?;
        let partial = sorted.partial_scalar()?;
        assert_eq!(
            partial.as_struct().field_by_idx(2),
            Some(Scalar::from(Some(1.0f64)))
        );
        assert_eq!(
            partial.as_struct().field_by_idx(3),
            Some(Scalar::from(Some(2.0f64)))
        );
        assert_eq!(
            sorted.finish()?,
            Scalar::bool(true, Nullability::NonNullable)
        );
    }

    Ok(())
}
