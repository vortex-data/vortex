// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential tests of sequence construction and its kernels against an exact `i128` model.

use num_traits::Bounded;
use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Sequence;

// Keep the `i128` oracle independent of the production code's 64-bit arithmetic.
fn widen(value: PValue) -> Option<i128> {
    vortex_array::match_each_pvalue!(
        value,
        uint: |v| { Some(i128::from(v)) },
        int: |v| { Some(i128::from(v)) },
        float: |_v| { None }
    )
}

fn narrow(value: i128, ptype: PType) -> Option<PValue> {
    match_each_integer_ptype!(ptype, |O| {
        num_traits::cast::<i128, O>(value).map(PValue::from)
    })
}

fn exact_value(base: PValue, multiplier: PValue, index: usize) -> Option<i128> {
    let base = widen(base)?;
    let multiplier = widen(multiplier)?;
    i128::try_from(index)
        .ok()
        .and_then(|index| multiplier.checked_mul(index))
        .and_then(|offset| base.checked_add(offset))
}

const VALUES: [PValue; 15] = [
    PValue::I8(i8::MIN),
    PValue::I8(i8::MAX),
    PValue::I16(-3),
    PValue::I32(-10),
    PValue::I32(0),
    PValue::I32(100),
    PValue::I64(i64::MIN),
    PValue::I64(i64::MAX),
    PValue::I64(1 << 62),
    PValue::I64(-255),
    PValue::U8(0),
    PValue::U8(u8::MAX),
    PValue::U32(1000),
    PValue::U64(u64::MAX),
    PValue::U64(1 << 63),
];

const PTYPES: [PType; 8] = [
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
];

const LENGTHS: [usize; 4] = [1, 2, 5, 300];

fn model(base: PValue, multiplier: PValue, ptype: PType, length: usize) -> Option<Vec<i128>> {
    let (min, max) = match_each_integer_ptype!(ptype, |P| {
        (
            <P as Bounded>::min_value().to_i128()?,
            <P as Bounded>::max_value().to_i128()?,
        )
    });

    (0..length)
        .map(|idx| exact_value(base, multiplier, idx).filter(|value| (min..=max).contains(value)))
        .collect()
}

fn values(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<i128>> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(primitive.ptype(), |P| {
        primitive
            .as_slice::<P>()
            .iter()
            .filter_map(|value| value.to_i128())
            .collect()
    }))
}

fn constant(value: i128, ptype: PType, len: usize) -> VortexResult<ArrayRef> {
    let value = narrow(value, ptype)
        .ok_or_else(|| vortex_err!("{value} is not representable in {ptype}"))?;
    let scalar = Scalar::try_new(
        DType::Primitive(ptype, NonNullable),
        Some(ScalarValue::Primitive(value)),
    )?;
    Ok(ConstantArray::new(scalar, len).into_array())
}

#[test]
fn sequence_kernels_match_exact_model() -> VortexResult<()> {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();

    let mut built = 0;
    for base in VALUES {
        for multiplier in VALUES {
            for ptype in PTYPES {
                for length in LENGTHS {
                    let expected = model(base, multiplier, ptype, length);
                    let array = Sequence::try_new(base, multiplier, ptype, NonNullable, length);
                    let case = format!("base {base:?}, step {multiplier:?}, {ptype}, len {length}");

                    let Some(expected) = expected else {
                        assert!(array.is_err(), "built an unrepresentable sequence: {case}");
                        continue;
                    };
                    let array = array
                        .map_err(|e| e.with_context(format!("representable sequence: {case}")))?
                        .into_array();
                    built += 1;

                    assert_eq!(values(&array, &mut ctx)?, expected, "canonical: {case}");

                    for idx in [0, length / 2, length - 1] {
                        let scalar = array.clone().execute_scalar(idx, &mut ctx)?;
                        assert_eq!(scalar.dtype(), array.dtype(), "scalar dtype: {case}");
                        assert_eq!(
                            widen(scalar.as_primitive().pvalue().unwrap()).unwrap(),
                            expected[idx],
                            "scalar at {idx}: {case}"
                        );
                    }

                    let indices = PrimitiveArray::from_iter((0..length as u64).rev()).into_array();
                    let taken = array.clone().take(indices)?;
                    let reversed = expected.iter().copied().rev().collect::<Vec<_>>();
                    assert_eq!(values(&taken, &mut ctx)?, reversed, "take: {case}");

                    let mask = Mask::from_iter((0..length).map(|idx| idx % 2 == 0));
                    let filtered = array.clone().filter(mask)?;
                    let every_other = expected.iter().copied().step_by(2).collect::<Vec<_>>();
                    assert_eq!(values(&filtered, &mut ctx)?, every_other, "filter: {case}");

                    if length > 2 {
                        let sliced = array.clone().slice(1..length - 1)?;
                        assert_eq!(
                            values(&sliced, &mut ctx)?,
                            expected[1..length - 1],
                            "slice: {case}"
                        );
                    }

                    let MinMaxResult { min, max } =
                        min_max(&array, &mut ctx, NumericalAggregateOpts::default())?
                            .expect("min_max of a non-empty sequence is not null");
                    assert_eq!(
                        widen(min.as_primitive().pvalue().unwrap()).unwrap(),
                        *expected.iter().min().unwrap(),
                        "min: {case}"
                    );
                    assert_eq!(
                        widen(max.as_primitive().pvalue().unwrap()).unwrap(),
                        *expected.iter().max().unwrap(),
                        "max: {case}"
                    );

                    let held = expected[length / 2];
                    for value in [Some(held), held.checked_add(1).filter(|v| *v != held)] {
                        let Some(value) = value.filter(|value| model_holds(ptype, *value)) else {
                            continue;
                        };
                        let compared = array
                            .clone()
                            .binary(constant(value, ptype, length)?, Operator::Eq)?
                            .execute::<BoolArray>(&mut ctx)?;
                        let bits = compared.to_bit_buffer();
                        let matches = (0..length).map(|idx| bits.value(idx)).collect::<Vec<_>>();
                        let expected_matches = expected
                            .iter()
                            .map(|element| *element == value)
                            .collect::<Vec<_>>();
                        assert_eq!(matches, expected_matches, "compare {value}: {case}");
                    }
                }
            }
        }
    }

    assert!(built > 1000, "only built {built} sequences");
    Ok(())
}

fn model_holds(ptype: PType, value: i128) -> bool {
    narrow(value, ptype).is_some()
}
