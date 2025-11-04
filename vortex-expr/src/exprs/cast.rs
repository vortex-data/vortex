// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::cast as compute_cast;
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata};
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog, VTable, vtable,
};

vtable!(Cast);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Eq)]
pub struct CastExpr {
    target: DType,
    child: ExprRef,
}

impl PartialEq for CastExpr {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.child.eq(&other.child)
    }
}

pub struct CastExprEncoding;

impl VTable for CastVTable {
    type Expr = CastExpr;
    type Encoding = CastExprEncoding;
    type Metadata = ProstMetadata<pb::CastOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("cast")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(CastExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::CastOpts {
            target: Some((&expr.target).into()),
        }))
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(CastExpr {
            target: expr.target.clone(),
            child: children[0].clone(),
        })
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 1 {
            vortex_bail!(
                "Cast expression must have exactly 1 child, got {}",
                children.len()
            );
        }
        let target: DType = metadata
            .target
            .as_ref()
            .ok_or_else(|| vortex_err!("missing target dtype in CastOpts"))?
            .try_into()?;
        Ok(CastExpr {
            target,
            child: children[0].clone(),
        })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let array = expr.child.evaluate(scope)?;
        compute_cast(&array, &expr.target).map_err(|e| {
            e.with_context(format!(
                "Failed to cast array of dtype {} to {}",
                array.dtype(),
                expr.target
            ))
        })
    }

    fn return_dtype(expr: &Self::Expr, _scope: &DType) -> VortexResult<DType> {
        Ok(expr.target.clone())
    }
}

impl CastExpr {
    pub fn new(child: ExprRef, target: DType) -> Self {
        Self { target, child }
    }

    pub fn new_expr(child: ExprRef, target: DType) -> ExprRef {
        Self::new(child, target).into_expr()
    }
}

impl DisplayAs for CastExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "cast({}, {})", self.child, self.target)
            }
            DisplayFormat::Tree => {
                write!(f, "Cast(target: {})", self.target)
            }
        }
    }
}

impl AnalysisExpr for CastExpr {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        self.child.max(catalog)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        self.child.min(catalog)
    }

    fn nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        self.child.nan_count(catalog)
    }

    fn field_path(&self) -> Option<FieldPath> {
        self.child.field_path()
    }
}

/// Creates an expression that casts values to a target data type.
///
/// Converts the input expression's values to the specified target type.
///
/// ```rust
/// # use vortex_dtype::{DType, Nullability, PType};
/// # use vortex_expr::{cast, root};
/// let expr = cast(root(), DType::Primitive(PType::I64, Nullability::NonNullable));
/// ```
pub fn cast(child: ExprRef, target: DType) -> ExprRef {
    CastExpr::new(child, target).into_expr()
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{ExprRef, Scope, cast, get_item, root, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            cast(root(), DType::Bool(Nullability::NonNullable))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = cast(root(), DType::Bool(Nullability::Nullable));
        let _ = expr.with_children(vec![root()]);
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: ExprRef = cast(
            get_item("a", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = expr.evaluate(&Scope::new(test_array)).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_eq!(expr.to_string(), "cast($.value, i64)");

        let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
        assert_eq!(expr2.to_string(), "cast($, bool?)");
    }
}
