// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::col;
use crate::get_item;
use crate::not;
use crate::root;
use crate::test_harness;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn is_strict() {
    assert!(
        not(root())
            .as_scalar()
            .is_some_and(|f| f.signature().is_strict())
    );
}

#[test]
fn preserves_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = BoolArray::from_iter([Some(false), None, Some(true)]).into_array();

    let result = input.apply(&not(root()))?;

    assert_arrays_eq!(
        result,
        BoolArray::from_iter([Some(true), None, Some(false)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn invert_booleans() {
    let mut ctx = SESSION.create_execution_ctx();
    let not_expr = not(root());
    let bools = BoolArray::from_iter([false, true, false, false, true, true]);
    let result = bools.into_array().apply(&not_expr).unwrap();
    assert_arrays_eq!(
        result,
        BoolArray::from_iter([true, false, true, true, false, false]),
        &mut ctx
    );
}

#[test]
fn test_display_order_of_operations() {
    let a = not(get_item("a", root()));
    let b = get_item("a", not(root()));
    assert_ne!(a.to_string(), b.to_string());
    assert_eq!(a.to_string(), "vortex.not($.a)");
    assert_eq!(b.to_string(), "vortex.not($).a");
}

#[test]
fn dtype() {
    let not_expr = not(root());
    let dtype = DType::Bool(Nullability::NonNullable);
    assert_eq!(
        not_expr.return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::NonNullable)
    );

    let dtype = test_harness::struct_dtype();
    assert_eq!(
        not(col("bool1")).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::NonNullable)
    );
}
