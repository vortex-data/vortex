// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential tests of sequence construction and its kernels against an exact `i128` model.

use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
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
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::Sequence;
use crate::SequenceArray;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

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
    let mut ctx = SESSION.create_execution_ctx();

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
                            .compare(constant(value, ptype, length)?, CompareOperator::Eq)?
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

/// Maximum length for Hegel-generated sequences; stays within `i8::MAX + 1` so all 8 integer
/// index types can address every row.
pub(crate) const HEGEL_MAX_LEN: usize = 64;

/// Draws a valid [`SequenceArray`] together with its exact `i128`-computed canonical
/// [`PrimitiveArray`] without ever rejecting.
///
/// Covers all 8 integer ptypes, both nullabilities, edge lengths (`1`, `2`, up to
/// [`HEGEL_MAX_LEN`]), negative steps on unsigned outputs, and unsigned steps exceeding `i64::MAX`.
pub(crate) fn draw_sequence(tc: &TestCase) -> VortexResult<(SequenceArray, PrimitiveArray)> {
    match tc.draw(gs::integers::<u8>().max_value(7)) {
        0 => draw_sequence_typed::<u8>(tc),
        1 => draw_sequence_typed::<u16>(tc),
        2 => draw_sequence_typed::<u32>(tc),
        3 => draw_sequence_typed::<u64>(tc),
        4 => draw_sequence_typed::<i8>(tc),
        5 => draw_sequence_typed::<i16>(tc),
        6 => draw_sequence_typed::<i32>(tc),
        _ => draw_sequence_typed::<i64>(tc),
    }
}

fn draw_sequence_typed<P>(tc: &TestCase) -> VortexResult<(SequenceArray, PrimitiveArray)>
where
    P: NativePType + Bounded + ToPrimitive + gs::Integer + Into<PValue> + 'static,
{
    let len = match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 => 1,
        1 => 2,
        _ => tc.draw(
            gs::integers::<usize>()
                .min_value(1)
                .max_value(HEGEL_MAX_LEN),
        ),
    };
    let base: P = tc.draw(gs::integers::<P>());
    let base_i128 = base.to_i128().vortex_expect("integer fits i128");
    let min_i128 = <P as Bounded>::min_value()
        .to_i128()
        .vortex_expect("min fits i128");
    let max_i128 = <P as Bounded>::max_value()
        .to_i128()
        .vortex_expect("max fits i128");

    let (multiplier, mult_i128): (PValue, i128) = if len == 1 {
        if tc.draw(gs::booleans()) {
            let step = tc.draw(gs::integers::<i64>());
            (PValue::from(step), i128::from(step))
        } else {
            let step = tc.draw(gs::integers::<u64>());
            (PValue::from(step), i128::from(step))
        }
    } else {
        let steps = (len - 1) as i128;
        let max_pos = u64::try_from((max_i128 - base_i128) / steps)
            .vortex_expect("positive step bound fits u64");
        let max_neg = u64::try_from((base_i128 - min_i128) / steps)
            .vortex_expect("negative step bound fits u64")
            .min(i64::MIN.unsigned_abs());

        if tc.draw(gs::booleans()) {
            let step = tc.draw(gs::integers::<u64>().max_value(max_pos));
            (PValue::from(step), i128::from(step))
        } else {
            let mag = tc.draw(gs::integers::<u64>().max_value(max_neg));
            let step = if mag == i64::MIN.unsigned_abs() {
                i64::MIN
            } else {
                -i64::try_from(mag).vortex_expect("magnitude fits i64")
            };
            (PValue::from(step), i128::from(step))
        }
    };

    let nullability = if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        NonNullable
    };

    let sequence = Sequence::try_new(base.into(), multiplier, P::PTYPE, nullability, len)?;
    let expected_values: Vec<P> = (0..len)
        .map(|idx| {
            let val = base_i128 + (idx as i128) * mult_i128;
            num_traits::cast::<i128, P>(val).vortex_expect("constructed sequence value fits P")
        })
        .collect();
    let canonical = PrimitiveArray::new(Buffer::from(expected_values), Validity::from(nullability));
    Ok((sequence, canonical))
}

/// Generated sequences match an exact `i128` oracle across bulk decode, `scalar_at`, serde,
/// scalar `compare`, and chained `slice`/`take`/`filter`/`mask` operations.
#[hegel::test(test_cases = 100, print_blob = true)]
fn test_sequence_matches_exact_model_and_ops(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let (sequence, canonical) = draw_sequence(&tc).vortex_expect("draw valid sequence");
    check_encoding_roundtrip_and_ops(
        &tc,
        &sequence.into_array(),
        &canonical.into_array(),
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
    .vortex_expect("sequence harness check");
}
