// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_compute::arrow::IntoArrow;
use vortex_compute::arrow::IntoVector;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_vector::Datum;
use vortex_vector::VectorOps;

use crate::ArrayRef;
use crate::compute::LikeOptions;
use crate::compute::like as like_compute;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::and;

/// Expression that performs SQL LIKE pattern matching.
pub struct Like;

impl VTable for Like {
    type Options = LikeOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.like")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LikeOpts {
                negated: instance.negated,
                case_insensitive: instance.case_insensitive,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::LikeOpts::decode(metadata)?;
        Ok(LikeOptions {
            negated: opts.negated,
            case_insensitive: opts.case_insensitive,
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("child"),
            1 => ChildName::from("pattern"),
            _ => unreachable!("Invalid child index {} for Like expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        if options.negated {
            write!(f, " not")?;
        }
        if options.case_insensitive {
            write!(f, " ilike ")?;
        } else {
            write!(f, " like ")?;
        }
        expr.child(1).fmt_sql(f)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        let pattern = &arg_dtypes[1];

        if !input.is_utf8() {
            vortex_bail!("LIKE expression requires UTF8 input dtype, got {}", input);
        }
        if !pattern.is_utf8() {
            vortex_bail!(
                "LIKE expression requires UTF8 pattern dtype, got {}",
                pattern
            );
        }

        Ok(DType::Bool(
            (input.is_nullable() || pattern.is_nullable()).into(),
        ))
    }

    fn evaluate(
        &self,
        options: &Self::Options,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let child = expr.child(0).evaluate(scope)?;
        let pattern = expr.child(1).evaluate(scope)?;
        like_compute(&child, &pattern, *options)
    }

    fn execute(&self, options: &Self::Options, args: ExecutionArgs) -> VortexResult<Datum> {
        let [child, pattern]: [Datum; _] = args
            .datums
            .try_into()
            .map_err(|_| vortex_err!("Wrong argument count"))?;

        let child = child.into_arrow()?;
        let pattern = pattern.into_arrow()?;

        let array = match (options.negated, options.case_insensitive) {
            (false, false) => arrow_string::like::like(child.as_ref(), pattern.as_ref()),
            (false, true) => arrow_string::like::ilike(child.as_ref(), pattern.as_ref()),
            (true, false) => arrow_string::like::nlike(child.as_ref(), pattern.as_ref()),
            (true, true) => arrow_string::like::nilike(child.as_ref(), pattern.as_ref()),
        }?;

        let vector = array.into_vector()?;
        if vector.len() == 1 && args.row_count != 1 {
            // Arrow returns a scalar datum result
            return Ok(Datum::Scalar(vector.scalar_at(0).into()));
        }

        Ok(Datum::Vector(array.into_vector()?.into()))
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        tracing::warn!("Computing validity for LIKE expression");
        let child_validity = expression.child(0).validity()?;
        let pattern_validity = expression.child(1).validity()?;
        Ok(Some(and(child_validity, pattern_validity)))
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }
}

pub fn like(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: false,
            case_insensitive: false,
        },
        [child, pattern],
    )
}

pub fn ilike(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: false,
            case_insensitive: true,
        },
        [child, pattern],
    )
}

pub fn not_like(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: true,
            case_insensitive: false,
        },
        [child, pattern],
    )
}

pub fn not_ilike(child: Expression, pattern: Expression) -> Expression {
    Like.new_expr(
        LikeOptions {
            negated: true,
            case_insensitive: true,
        },
        [child, pattern],
    )
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::like::like;
    use crate::expr::exprs::like::not_ilike;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::not::not;
    use crate::expr::exprs::root::root;

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(&bools.to_array())
                .unwrap()
                .to_bool()
                .bit_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn dtype() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let like_expr = like(root(), lit("%test%"));
        assert_eq!(
            like_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = like(get_item("name", root()), lit("%john%"));
        assert_eq!(expr.to_string(), "$.name like \"%john%\"");

        let expr2 = not_ilike(root(), lit("test*"));
        assert_eq!(expr2.to_string(), "$ not ilike \"test*\"");
    }
}
