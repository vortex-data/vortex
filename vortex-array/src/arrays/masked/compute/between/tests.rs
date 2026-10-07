// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;
use crate::validity::Validity;

#[test]
fn row_bounds_keep_mask_and_kleene_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = MaskedArray::try_new(
        buffer![0i32, 1, 2, 3].into_array(),
        Validity::from_iter([true, false, true, true]),
    )?
    .into_array();
    let result = array
        .between(
            PrimitiveArray::from_option_iter([Some(0i32), Some(0), None, Some(0)]).into_array(),
            buffer![1i32, 3, 1, 4].into_array(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        )?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(true), None, Some(false), Some(true)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn null_bound_keeps_false_and_masked_rows() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = MaskedArray::try_new(
        buffer![0i32, 1, 2].into_array(),
        Validity::from_iter([true, false, true]),
    )?
    .into_array();
    let result = array
        .between(
            ConstantArray::new(
                Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                3,
            )
            .into_array(),
            ConstantArray::new(1i32, 3).into_array(),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        )?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([None, None, Some(false)]),
        &mut ctx
    );
    Ok(())
}
