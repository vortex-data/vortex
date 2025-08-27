// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::{ArrayRef, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::{DType, match_each_float_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog, VTable, vtable,
};

vtable!(Literal);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LiteralExpr {
    value: Scalar,
}

pub struct LiteralExprEncoding;

impl VTable for LiteralVTable {
    type Expr = LiteralExpr;
    type Encoding = LiteralExprEncoding;
    type Metadata = ProstMetadata<pb::LiteralOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("literal")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(LiteralExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::LiteralOpts {
            value: Some((&expr.value).into()),
        }))
    }

    fn children(_expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![]
    }

    fn with_children(expr: &Self::Expr, _children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(expr.clone())
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            vortex_bail!(
                "Literal expression does not have children, got: {:?}",
                children
            );
        }
        let value: Scalar = metadata
            .value
            .as_ref()
            .ok_or_else(|| vortex_err!("Literal metadata missing value"))?
            .try_into()?;
        Ok(LiteralExpr::new(value))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(expr.value.clone(), scope.len()).into_array())
    }

    fn return_dtype(expr: &Self::Expr, _scope: &DType) -> VortexResult<DType> {
        Ok(expr.value.dtype().clone())
    }
}

impl LiteralExpr {
    pub fn new(value: impl Into<Scalar>) -> Self {
        Self {
            value: value.into(),
        }
    }

    pub fn new_expr(value: impl Into<Scalar>) -> ExprRef {
        Self::new(value).into_expr()
    }

    pub fn value(&self) -> &Scalar {
        &self.value
    }

    pub fn maybe_from(expr: &ExprRef) -> Option<&LiteralExpr> {
        expr.as_opt::<LiteralVTable>()
    }
}

impl DisplayAs for LiteralExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "{}", self.value)
            }
            DisplayFormat::Tree => {
                write!(
                    f,
                    "Literal(value: {}, dtype: {})",
                    self.value,
                    self.value.dtype()
                )
            }
        }
    }
}

impl AnalysisExpr for LiteralExpr {
    fn max(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        Some(lit(self.value.clone()))
    }

    fn min(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        Some(lit(self.value.clone()))
    }

    fn nan_count(&self, _catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        // The NaNCount for a non-float literal is not defined.
        // For floating point types, the NaNCount is 1 for lit(NaN), and 0 otherwise.
        let value = self.value.as_primitive_opt()?;
        if !value.ptype().is_float() {
            return None;
        }

        match_each_float_ptype!(value.ptype(), |T| {
            match value.typed_value::<T>() {
                None => Some(lit(0u64)),
                Some(value) if value.is_nan() => Some(lit(1u64)),
                _ => Some(lit(0u64)),
            }
        })
    }
}

/// Create a new `Literal` expression from a type that coerces to `Scalar`.
///
///
/// ## Example usage
///
/// ```
/// use vortex_array::arrays::PrimitiveArray;
/// use vortex_dtype::Nullability;
/// use vortex_expr::{lit, LiteralVTable};
/// use vortex_scalar::Scalar;
///
/// let number = lit(34i32);
///
/// let literal = number.as_::<LiteralVTable>();
/// assert_eq!(literal.value(), &Scalar::primitive(34i32, Nullability::NonNullable));
/// ```
pub fn lit(value: impl Into<Scalar>) -> ExprRef {
    LiteralExpr::new(value.into()).into_expr()
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType, StructFields};
    use vortex_scalar::Scalar;

    use crate::{lit, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        assert_eq!(
            lit(10).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            lit(i64::MAX).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_eq!(
            lit(true).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            lit(Scalar::null(DType::Bool(Nullability::Nullable)))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );

        let sdtype = DType::Struct(
            StructFields::new(
                ["dog", "cat"].into(),
                vec![
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );
        assert_eq!(
            lit(Scalar::struct_(
                sdtype.clone(),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .return_dtype(&dtype)
            .unwrap(),
            sdtype
        );
    }
}
