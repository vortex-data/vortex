// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_error::VortexResult;

use crate::FoR;

#[rstest]
#[case(100i8, -20i8, 20i8)]
#[case(-100, -20, 20)]
#[case(1, i8::MIN, i8::MAX)]
#[case(1, i8::MAX, i8::MAX)]
#[case(-1, i8::MIN, i8::MIN)]
#[case(10, 20, -20)]
fn signed_wraparound(
    #[case] reference: i8,
    #[case] lower: i8,
    #[case] upper: i8,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values = [
        Some(i8::MIN),
        Some(-20),
        Some(-1),
        Some(0),
        Some(1),
        Some(20),
        Some(i8::MAX),
        None,
    ];
    let encoded = PrimitiveArray::from_option_iter(
        values.map(|value| value.map(|value| value.wrapping_sub(reference))),
    )
    .into_array();
    let array = FoR::try_new(encoded, Scalar::from(reference))?.into_array();
    let result = array
        .between(
            ConstantArray::new(
                Scalar::primitive(lower, Nullability::Nullable),
                values.len(),
            )
            .into_array(),
            ConstantArray::new(upper, values.len()).into_array(),
            BetweenOptions {
                lower_strict,
                upper_strict,
            },
        )?
        .execute::<BoolArray>(&mut ctx)?;
    let expected = BoolArray::from_iter(values.map(|value| {
        value.map(|value| {
            (if lower_strict.is_strict() {
                value > lower
            } else {
                value >= lower
            }) && (if upper_strict.is_strict() {
                value < upper
            } else {
                value <= upper
            })
        })
    }));
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn unsigned_wraparound() -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values = [0u64, 1, u64::MAX - 1, u64::MAX];
    let array = FoR::try_new(
        PrimitiveArray::from_iter(values.map(|value| value.wrapping_sub(1))).into_array(),
        Scalar::from(1u64),
    )?
    .into_array();
    let result = array
        .between(
            ConstantArray::new(0u64, values.len()).into_array(),
            ConstantArray::new(1u64, values.len()).into_array(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, true, false, false]),
        &mut ctx
    );
    Ok(())
}
