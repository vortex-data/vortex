// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::stats::Stat;
use crate::match_each_float_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

fn lit(value: impl Into<Scalar>) -> Expression {
    Literal.new_expr(value.into(), [])
}

/// Expression that represents a literal scalar value.
#[derive(Clone)]
pub struct Literal;

impl ScalarFnVTable for Literal {
    type Options = Scalar;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.literal")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LiteralOpts {
                value: Some(instance.into()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let ops = pb::LiteralOpts::decode(_metadata)?;
        Scalar::from_proto(
            ops.value
                .as_ref()
                .ok_or_else(|| vortex_err!("Literal metadata missing value"))?,
            session,
        )
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _instance: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(
        &self,
        scalar: &Scalar,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", scalar)
    }

    fn return_dtype(&self, options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(options.dtype().clone())
    }

    fn execute(
        &self,
        scalar: &Scalar,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(scalar.clone(), args.row_count()).into_array())
    }

    fn stat_expression(
        &self,
        scalar: &Scalar,
        _expr: &Expression,
        stat: Stat,
        _catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // NOTE(ngates): we return incorrect `1` values for counts here since we don't have
        //  row-count information. We could resolve this in the future by introducing a `count()`
        //  expression that evaluates to the row count of the provided scope. But since this is
        //  only currently used for pruning, it doesn't change the outcome.

        match stat {
            Stat::Min | Stat::Max => Some(lit(scalar.clone())),
            Stat::IsConstant => Some(lit(true)),
            Stat::NaNCount => {
                // The NaNCount for a non-float literal is not defined.
                // For floating point types, the NaNCount is 1 for lit(NaN), and 0 otherwise.
                let value = scalar.as_primitive_opt()?;
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
                if scalar.is_null() {
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

    fn validity(
        &self,
        scalar: &Scalar,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(lit(scalar.is_valid())))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _instance: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::lit;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;

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
