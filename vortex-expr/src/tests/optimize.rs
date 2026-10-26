// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayExprExt;
use crate::Expression;
use crate::and;
use crate::cast;
use crate::eq;
use crate::get_item;
use crate::lit;
use crate::lt_eq;
use crate::or;
use crate::root;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[test]
fn optimize_or_chain_correctness() -> VortexResult<()> {
    let expr = or(
        eq(get_item("x", root()), lit(1i32)),
        eq(get_item("x", root()), lit(2i32)),
    );
    let scope = DType::Struct(
        StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    );
    let optimized = expr.bind(&scope)?.optimize_recursive()?;

    let data =
        StructArray::from_fields(&[("x", buffer![0i32, 1, 2, 3].into_array())])?.into_array();
    let actual = data.apply_bound(&optimized)?;
    assert_arrays_eq!(
        actual,
        BoolArray::from_iter([false, true, true, false]),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn optimize_folds_cast_of_literal_in_comparison() -> VortexResult<()> {
    let expr = lt_eq(
        get_item("x", root()),
        cast(
            lit(3i32),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        ),
    );
    let scope = DType::Struct(
        StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::F64, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    );
    let optimized = expr.bind(&scope)?.optimize_recursive()?;

    // Prune rules pattern-match a bare Literal on the comparison RHS; a cast wrapper
    // silently disables pruning.
    let rhs = optimized
        .child(1)
        .as_opt::<Literal>()
        .ok_or_else(|| vortex_err!("expected a bare literal RHS, got {optimized}"))?;
    assert_eq!(rhs, &Scalar::primitive(3.0f64, Nullability::NonNullable));
    Ok(())
}

#[rstest]
#[case::and_annihilator(and(root(), lit(false)), DType::Bool(Nullability::Nullable))]
#[case::or_annihilator(or(root(), lit(true)), DType::Bool(Nullability::Nullable))]
#[case::and_nullable_identity(
    and(root(), lit(Scalar::from(Some(true)))),
    DType::Bool(Nullability::NonNullable)
)]
#[case::or_nullable_identity(
    or(root(), lit(Scalar::from(Some(false)))),
    DType::Bool(Nullability::NonNullable)
)]
fn literal_simplification_preserves_result_dtype(
    #[case] expr: Expression,
    #[case] scope: DType,
) -> VortexResult<()> {
    let original = expr.bind(&scope)?;
    let optimized = expr.bind(&scope)?.optimize_recursive()?;
    assert_eq!(optimized.dtype(), original.dtype());

    Ok(())
}
