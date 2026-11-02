// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

use crate::arrays::shared::Shared;
use crate::arrays::shared::SharedArrayExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_onpair::initialize(&session);
    session
});

#[test]
fn shared_array_caches_on_canonicalize() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
    let shared = Shared::new(array);

    let mut ctx = SESSION.create_execution_ctx();

    let first = shared.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

    // Second call should return cached without invoking the closure.
    let second = shared.get_or_compute(|_| panic!("should not execute twice"))?;

    assert!(ArrayRef::ptr_eq(&first, &second));
    assert_arrays_eq!(first, PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx);

    Ok(())
}

#[rstest]
#[case(Operator::Eq)]
#[case(Operator::NotEq)]
#[case(Operator::Gt)]
#[case(Operator::Gte)]
#[case(Operator::Lt)]
#[case(Operator::Lte)]
#[cfg_attr(miri, ignore)]
fn shared_empty_comparison_keeps_values_encoded(
    #[case] operator: Operator,
    #[values(false, true)] reversed: bool,
    #[values(false, true)] binary: bool,
) -> VortexResult<()> {
    let dtype = if binary {
        DType::Binary(Nullability::Nullable)
    } else {
        DType::Utf8(Nullability::Nullable)
    };
    let input = VarBinArray::from_iter([Some(""), None, Some("abc")], dtype).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?.into_array();
    let shared = Shared::new(encoded.clone());
    let empty = input.probe().execute_scalar(0, &mut ctx)?;
    let constant = ConstantArray::new(empty, input.len()).into_array();
    let (result, expected) = if reversed {
        (
            constant.binary(shared.clone().into_array(), operator)?,
            constant.binary(input, operator)?,
        )
    } else {
        (
            shared
                .clone()
                .into_array()
                .binary(constant.clone(), operator)?,
            input.binary(constant, operator)?,
        )
    };
    assert_arrays_eq!(result, expected, &mut ctx);
    assert!(
        ArrayRef::ptr_eq(shared.current_array_ref(), &encoded),
        "comparing with an empty string must not decode the shared dictionary"
    );
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_nonempty_ordering_caches_values() -> VortexResult<()> {
    let input = VarBinArray::from_iter(
        [Some(""), None, Some("abc")],
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?.into_array();
    let shared = Shared::new(encoded.clone());
    let expected = BoolArray::from_iter([Some(false), None, Some(true)]);
    let comparison = shared.clone().into_array().binary(
        ConstantArray::new("a", input.len()).into_array(),
        Operator::Gt,
    )?;
    assert_arrays_eq!(comparison, expected, &mut ctx);
    let cached = shared.current_array_ref().clone();
    assert!(!ArrayRef::ptr_eq(&cached, &encoded));
    assert_arrays_eq!(comparison, expected, &mut ctx);
    assert!(ArrayRef::ptr_eq(shared.current_array_ref(), &cached));
    Ok(())
}
