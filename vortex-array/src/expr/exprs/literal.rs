// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::match_each_float_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::expr::ChildName;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::ExpressionView;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::stats::Stat;

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

    fn stat_expression(
        &self,
        expr: &ExpressionView<Self>,
        stat: Stat,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // NOTE(ngates): we return incorrect `1` values for counts here since we don't have
        //  row-count information. We could resolve this in the future by introducing a `count()`
        //  expression that evaluates to the row count of the provided scope. But since this is
        //  only currently used for pruning, it doesn't change the outcome.

        match stat {
            Stat::Min | Stat::Max => Some(lit(expr.data().clone())),
            Stat::IsConstant => Some(lit(true)),
            Stat::NaNCount => {
                // The NaNCount for a non-float literal is not defined.
                // For floating point types, the NaNCount is 1 for lit(NaN), and 0 otherwise.
                let value = expr.data().as_primitive_opt()?;
                if !value.ptype().is_float() {
                    return None;
                }

                match_each_float_ptype!(value.ptype(), |T| {
                    if value.typed_value::<T>().is_some_and(|v| v.is_nan()) {
                        Some(lit(1u64))
                    } else {
                        Some(lit(0u64))
                    }
                })
            }
            Stat::NullCount => {
                if expr.data().is_null() {
                    Some(lit(1u64))
                } else {
                    Some(lit(0u64))
                }
            }
            Stat::IsSorted | Stat::IsStrictSorted | Stat::Sum | Stat::UncompressedSizeInBytes => {
                None
            }
        }
    }

    fn is_null_sensitive(&self, _instance: &Self::Instance) -> bool {
        false
    }

    fn is_fallible(&self, _instance: &Self::Instance) -> bool {
        false
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
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::StructFields;
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
