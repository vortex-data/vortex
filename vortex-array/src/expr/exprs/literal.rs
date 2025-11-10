// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_dtype::{DType, match_each_float_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;

use crate::arrays::ConstantArray;
use crate::expr::{ChildName, ExprId, Expression, ExpressionView, StatsCatalog, VTable, VTableExt};
use crate::{Array, ArrayRef, IntoArray};

/// Expression that represents a literal scalar value.
pub struct Literal;

impl VTable for Literal {
    type Instance = Scalar;

    fn id(&self) -> ExprId {
        ExprId::new_ref("vortex.literal")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LiteralOpts {
                value: Some(instance.as_ref().into()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let ops = pb::LiteralOpts::decode(metadata)?;
        Ok(Some(
            ops.value
                .as_ref()
                .ok_or_else(|| vortex_err!("Literal metadata missing value"))?
                .try_into()?,
        ))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if !expr.children().is_empty() {
            vortex_bail!(
                "Literal expression does not have children, got: {:?}",
                expr.children()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", expr.data())
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", instance)
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, _scope: &DType) -> VortexResult<DType> {
        Ok(expr.data().dtype().clone())
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(expr.data().clone(), scope.len()).into_array())
    }

    fn stat_max(
        &self,
        expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        Some(lit(expr.data().clone()))
    }

    fn stat_min(
        &self,
        expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        Some(lit(expr.data().clone()))
    }

    fn stat_nan_count(
        &self,
        expr: &ExpressionView<Self>,
        _catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        // The NaNCount for a non-float literal is not defined.
        // For floating point types, the NaNCount is 1 for lit(NaN), and 0 otherwise.
        let value = expr.data().as_primitive_opt()?;
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
/// use vortex_array::expr::{lit, Literal};
/// use vortex_scalar::Scalar;
///
/// let number = lit(34i32);
///
/// let literal = number.as_::<Literal>();
/// assert_eq!(literal.data(), &Scalar::primitive(34i32, Nullability::NonNullable));
/// ```
pub fn lit(value: impl Into<Scalar>) -> Expression {
    Literal.new_expr(value.into(), [])
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType, StructFields};
    use vortex_scalar::Scalar;

    use super::lit;
    use crate::expr::test_harness;

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
