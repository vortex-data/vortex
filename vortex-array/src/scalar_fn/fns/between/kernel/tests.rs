// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::between_compare;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::patches::Patches;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::StrictComparison;

#[rstest]
fn strings_use_comparison_kernels(
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] lower_strict: StrictComparison,
    #[values(StrictComparison::Strict, StrictComparison::NonStrict)] upper_strict: StrictComparison,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = [Some(""), Some("a"), Some("b"), Some("c"), None];
    let array = VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array();
    let lower = ConstantArray::new("", values.len()).into_array();
    let upper = ConstantArray::new("b", values.len()).into_array();
    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };
    let result = between_compare(array.as_::<VarBin>(), &lower, &upper, &options, &mut ctx)?
        .ok_or_else(|| vortex_err!("expected varbin comparison kernels"))?;
    let expected = BoolArray::from_iter(values.map(|value| {
        value.map(|value| {
            (if lower_strict.is_strict() {
                !value.is_empty()
            } else {
                true
            }) && (if upper_strict.is_strict() {
                value < "b"
            } else {
                value <= "b"
            })
        })
    }));
    assert_arrays_eq!(result, expected, &mut ctx);
    let result = array
        .between(lower, upper, options)?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn unhandled_comparison_declines() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DType::Utf8(Nullability::NonNullable);
    let array = VarBinArray::from_iter([Some("a"), Some("b")], dtype.clone()).into_array();
    let lower = VarBinArray::from_iter([Some(""), Some("c")], dtype).into_array();
    let upper = ConstantArray::new("b", 2).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    };
    assert!(between_compare(array.as_::<VarBin>(), &lower, &upper, &options, &mut ctx)?.is_none());
    let result = array
        .between(lower, upper, options)?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(result, BoolArray::from_iter([true, false]), &mut ctx);
    Ok(())
}

#[test]
fn patched_comparisons_apply_exceptions() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let patches = Patches::new(
        4,
        0,
        buffer![1u16, 3].into_array(),
        buffer![5i32, -1].into_array(),
        None,
    )?;
    let array =
        Patched::from_array_and_patches(buffer![0i32, 1, 2, 3].into_array(), &patches, &mut ctx)?
            .into_array();
    let lower = ConstantArray::new(0i32, 4).into_array();
    let upper = ConstantArray::new(3i32, 4).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::Strict,
    };
    let result = between_compare(array.as_::<Patched>(), &lower, &upper, &options, &mut ctx)?
        .ok_or_else(|| vortex_err!("expected patched comparison kernels"))?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, false, true, false]),
        &mut ctx
    );
    let result = array
        .between(lower, upper, options)?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, false, true, false]),
        &mut ctx
    );
    Ok(())
}
