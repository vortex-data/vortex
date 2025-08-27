// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_expr::display::{DisplayAs, DisplayFormat};
use vortex_expr::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable,
};

vtable!(RowIdx);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowIdxExpr;

impl AnalysisExpr for RowIdxExpr {}

impl DisplayAs for RowIdxExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => write!(f, "#row_idx"),
            DisplayFormat::Tree => write!(f, "RowIdx"),
        }
    }
}

#[derive(Clone)]
pub struct RowIdxExprEncoding;

impl VTable for RowIdxVTable {
    type Expr = RowIdxExpr;
    type Encoding = RowIdxExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("vortex.row_idx")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(RowIdxExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        // Serializable, but with no metadata
        Some(EmptyMetadata)
    }

    fn children(_expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![]
    }

    fn with_children(expr: &Self::Expr, _children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(expr.clone())
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            vortex_bail!(
                "RowIdxExpr does not expect any children, got {}",
                children.len()
            );
        }
        Ok(RowIdxExpr)
    }

    fn evaluate(_expr: &Self::Expr, _scope: &Scope) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "RowIdxExpr should not be evaluated directly, use it in the context of a Vortex scan and it will be substituted for a row index array"
        );
    }

    fn return_dtype(_expr: &Self::Expr, _scope: &DType) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::U64, Nullability::NonNullable))
    }
}

pub fn row_idx() -> ExprRef {
    RowIdxExpr.into_expr()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::EmptyMetadata;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_expr::{ExprRef, IntoExpr, VTable};

    use crate::layouts::row_idx::{RowIdxExpr, RowIdxExprEncoding, RowIdxVTable, row_idx};

    #[test]
    fn test_row_idx_expr_creation() {
        let expr = row_idx();
        assert!(expr.is::<RowIdxVTable>());
    }

    #[test]
    fn test_vtable_id() {
        let encoding = RowIdxExprEncoding;
        let id = RowIdxVTable::id(&encoding);
        assert_eq!(id.as_ref(), "vortex.row_idx");
    }

    #[test]
    fn test_vtable_encoding() {
        let expr = RowIdxExpr;
        let encoding_ref = RowIdxVTable::encoding(&expr);

        // Check that the encoding ref is the same instance
        let encoding_ref2 = RowIdxVTable::encoding(&expr);
        assert!(std::ptr::eq(
            encoding_ref.as_ref() as *const _,
            encoding_ref2.as_ref() as *const _
        ));
    }

    #[test]
    fn test_vtable_metadata() {
        let expr = RowIdxExpr;
        let metadata = RowIdxVTable::metadata(&expr);
        assert!(metadata.is_some());
        // Just check that we get EmptyMetadata back (can't compare directly)
        let _empty = metadata.unwrap();
    }

    #[test]
    fn test_vtable_children() {
        let expr = RowIdxExpr;
        let children = RowIdxVTable::children(&expr);
        assert!(children.is_empty());
    }

    #[test]
    fn test_vtable_with_children() {
        let expr = RowIdxExpr;

        // Should succeed with empty children
        let result = RowIdxVTable::with_children(&expr, vec![]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expr);

        // Should also succeed with non-empty children (it ignores them and returns the same expr)
        let dummy_expr = row_idx();
        let result = RowIdxVTable::with_children(&expr, vec![dummy_expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expr);
    }

    #[test]
    fn test_vtable_build_success() {
        let encoding = RowIdxExprEncoding;
        let metadata = EmptyMetadata;
        let result = RowIdxVTable::build(&encoding, &metadata, vec![]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RowIdxExpr);
    }

    #[test]
    fn test_vtable_build_with_children_fails() {
        let encoding = RowIdxExprEncoding;
        let metadata = EmptyMetadata;
        let dummy_expr = row_idx();
        let result = RowIdxVTable::build(&encoding, &metadata, vec![dummy_expr]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("does not expect any children"));
        assert!(err_msg.contains("got 1"));
    }

    #[test]
    fn test_vtable_build_with_multiple_children_fails() {
        let encoding = RowIdxExprEncoding;
        let metadata = EmptyMetadata;
        let dummy_expr1 = row_idx();
        let dummy_expr2 = row_idx();
        let result = RowIdxVTable::build(&encoding, &metadata, vec![dummy_expr1, dummy_expr2]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("does not expect any children"));
        assert!(err_msg.contains("got 2"));
    }

    #[test]
    fn test_vtable_evaluate_fails() {
        use vortex_array::IntoArray;
        use vortex_array::arrays::PrimitiveArray;
        use vortex_expr::Scope;

        let expr = RowIdxExpr;
        // Create a dummy array for the scope
        let array = PrimitiveArray::from_iter([0u64, 1, 2]).into_array();
        let scope = Scope::new(array);
        let result = RowIdxVTable::evaluate(&expr, &scope);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("should not be evaluated directly"));
        assert!(err_msg.contains("context of a Vortex scan"));
    }

    #[test]
    fn test_vtable_return_dtype() {
        let expr = RowIdxExpr;
        let scope = DType::Primitive(PType::U32, Nullability::Nullable);
        let result = RowIdxVTable::return_dtype(&expr, &scope);
        assert!(result.is_ok());
        let dtype = result.unwrap();
        assert_eq!(
            dtype,
            DType::Primitive(PType::U64, Nullability::NonNullable)
        );
    }

    #[rstest]
    #[case(DType::Primitive(PType::U8, Nullability::Nullable))]
    #[case(DType::Primitive(PType::I64, Nullability::NonNullable))]
    #[case(DType::Primitive(PType::F32, Nullability::Nullable))]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_vtable_return_dtype_with_various_scopes(#[case] scope_dtype: DType) {
        let expr = RowIdxExpr;
        let result = RowIdxVTable::return_dtype(&expr, &scope_dtype);
        assert!(result.is_ok());
        // Row index dtype should always be U64 NonNullable regardless of scope
        assert_eq!(
            result.unwrap(),
            DType::Primitive(PType::U64, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_row_idx_function_creates_correct_expr() {
        let expr = row_idx();

        // Verify it's the right type
        assert!(expr.is::<RowIdxVTable>());

        // Verify it has no children
        let children = RowIdxVTable::children(&RowIdxExpr);
        assert_eq!(children.len(), 0);

        // Verify metadata is present
        let metadata = RowIdxVTable::metadata(&RowIdxExpr);
        assert!(metadata.is_some());
    }

    #[test]
    fn test_expr_into_expr_conversion() {
        let expr = RowIdxExpr;
        let expr_ref: ExprRef = expr.into_expr();
        assert!(expr_ref.is::<RowIdxVTable>());
    }
}
