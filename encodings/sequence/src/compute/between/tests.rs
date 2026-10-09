// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_error::VortexResult;

use crate::Sequence;

#[rstest]
#[case(-5i64, 2i64, -1i64, 3i64)]
#[case(5, -2, -1, 3)]
#[case(1, 0, 0, 2)]
#[case(5, 0, 0, 2)]
#[case(-5, 2, 3, -1)]
#[case(i64::MIN, 1, i64::MIN, i64::MIN + 2)]
#[case(i64::MAX - 5, 1, i64::MAX - 2, i64::MAX)]
fn signed_sequences(
    #[case] base: i64,
    #[case] step: i64,
    #[case] lower: i64,
    #[case] upper: i64,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let array = Sequence::try_new_typed(base, step, Nullability::NonNullable, 6)?.into_array();
    let result = array
        .between(
            ConstantArray::new(Scalar::primitive(lower, Nullability::Nullable), 6).into_array(),
            ConstantArray::new(upper, 6).into_array(),
            BetweenOptions {
                lower_strict,
                upper_strict,
            },
        )?
        .execute::<BoolArray>(&mut ctx)?;
    let expected = BoolArray::from_iter((0..6).map(|index| {
        let value = base + index * step;
        Some(
            (if lower_strict.is_strict() {
                value > lower
            } else {
                value >= lower
            }) && (if upper_strict.is_strict() {
                value < upper
            } else {
                value <= upper
            }),
        )
    }));
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case(u64::MAX, -1i64, u64::MAX - 3, u64::MAX)]
#[case(
    0x7fff_ffff_ffff_ffffu64,
    1,
    0x8000_0000_0000_0000,
    0x8000_0000_0000_0002
)]
fn unsigned_sequences(
    #[case] base: u64,
    #[case] step: i64,
    #[case] lower: u64,
    #[case] upper: u64,
) -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let array = Sequence::try_new(
        PValue::from(base),
        PValue::from(step),
        PType::U64,
        Nullability::NonNullable,
        6,
    )?
    .into_array();
    let result = array
        .between(
            ConstantArray::new(lower, 6).into_array(),
            ConstantArray::new(upper, 6).into_array(),
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        )?
        .execute::<BoolArray>(&mut ctx)?;
    let expected = BoolArray::from_iter((0..6).map(|index| {
        let value = i128::from(base) + i128::from(index) * i128::from(step);
        value > i128::from(lower) && value <= i128::from(upper)
    }));
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}
