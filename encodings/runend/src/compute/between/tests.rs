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
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::RunEnd;

#[rstest]
fn constant_bounds_with_offset(
    #[values(0, 1, 3)] offset: usize,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let values = [
        Some(-1),
        Some(-1),
        Some(0),
        Some(0),
        Some(1),
        Some(1),
        None,
        None,
        Some(2),
        Some(2),
    ];
    let len = 10 - offset;
    let array = RunEnd::try_new_offset_length(
        PrimitiveArray::from_iter([2u32, 4, 6, 8, 10].into_iter().skip(offset / 2)).into_array(),
        PrimitiveArray::from_option_iter(
            [Some(-1i32), Some(0), Some(1), None, Some(2)]
                .into_iter()
                .skip(offset / 2),
        )
        .into_array(),
        offset,
        len,
        &mut ctx,
    )?;
    let lower =
        ConstantArray::new(Scalar::primitive(0i32, Nullability::Nullable), len).into_array();
    let upper = ConstantArray::new(1i32, len).into_array();
    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };
    let expected = BoolArray::from_iter(values[offset..].iter().map(|value| {
        value.map(|value| {
            let lower_matches = if lower_strict.is_strict() {
                value > 0
            } else {
                value >= 0
            };
            let upper_matches = if upper_strict.is_strict() {
                value < 1
            } else {
                value <= 1
            };
            lower_matches && upper_matches
        })
    }));
    let result =
        <RunEnd as BetweenKernel>::between(array.as_view(), &lower, &upper, &options, &mut ctx)?
            .ok_or_else(|| vortex_err!("expected run-end between kernel"))?;
    assert_arrays_eq!(result, expected, &mut ctx);
    let result = array
        .into_array()
        .between(lower, upper, options)?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn row_bounds_decline() -> VortexResult<()> {
    let session = array_session();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let array = RunEnd::try_new(
        buffer![2u32, 4].into_array(),
        buffer![0i32, 1].into_array(),
        &mut ctx,
    )?;
    assert!(
        <RunEnd as BetweenKernel>::between(
            array.as_view(),
            &buffer![-1i32, 0, 1, 2].into_array(),
            &ConstantArray::new(2i32, 4).into_array(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
            &mut ctx,
        )?
        .is_none()
    );
    Ok(())
}
