// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::tests::SESSION;

/// A one-off or retained probe over `array`, the retained one living in `retained`.
fn probe_for<'a>(
    array: &'a ArrayRef,
    retained: &'a mut Option<RepeatedArrayProbe>,
    repeated: bool,
) -> ArrayProbe<'a> {
    if repeated {
        retained.insert(array.repeated_probe()).as_probe()
    } else {
        array.probe()
    }
}

#[rstest]
fn sliced_strings_and_null_runs(#[values(false, true)] repeated: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = RunEnd::try_new_offset_length(
        buffer![2u8, 5, 9].into_array(),
        VarBinViewArray::from_iter(
            [Some("first"), None, Some("last")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array(),
        1,
        7,
        &mut ctx,
    )?
    .into_array();
    let mut retained = None;
    let mut probe = probe_for(&array, &mut retained, repeated);
    let mut actual = builder_with_capacity_in(array.dtype(), 5, ctx.allocator());
    for index in [6, 0, 1, 3, 4] {
        actual.append_scalar(&probe.execute_scalar(index, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        VarBinViewArray::from_iter(
            [Some("last"), Some("first"), None, None, Some("last")],
            DType::Utf8(Nullability::Nullable)
        ),
        &mut ctx
    );
    assert!(probe.execute_scalar(array.len(), &mut ctx).is_err());
    Ok(())
}

#[test]
fn empty_probe_checks_bounds() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = RunEnd::try_new(
        Buffer::<u32>::empty().into_array(),
        Buffer::<u32>::empty().into_array(),
        &mut ctx,
    )?
    .into_array();
    assert!(array.probe().execute_scalar(0, &mut ctx).is_err());
    assert!(array.repeated_probe().execute_scalar(0, &mut ctx).is_err());
    Ok(())
}

#[test]
fn lazy_value_validity_only_evaluates_requested_runs() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let numerators = buffer![1u32, 1].into_array();
    let validity = numerators
        .binary(buffer![1u32, 0].into_array(), Operator::Div)?
        .binary(numerators, Operator::Eq)?;
    let values = PrimitiveArray::new(buffer![42u32, 99], Validity::Array(validity));
    let array =
        RunEnd::try_new(buffer![4u32, 8].into_array(), values.into_array(), &mut ctx)?.into_array();
    let expected = array.execute_scalar(0, &mut ctx)?;
    let mut probe = array.repeated_probe();
    assert_eq!(probe.execute_scalar(0, &mut ctx)?, expected);
    assert_eq!(probe.execute_scalar(3, &mut ctx)?, expected);
    assert!(probe.execute_scalar(4, &mut ctx).is_err());
    Ok(())
}
