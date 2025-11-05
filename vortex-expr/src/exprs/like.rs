// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::compute::{LikeOptions, like as like_compute};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_proto::expr as pb;

use crate::{ChildName, ExprId, Expression, ExpressionView, VTable, VTableExt};

/// Expression that performs SQL LIKE pattern matching.
pub struct Like;

impl VTable for Like {
    type Instance = LikeOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.like")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LikeOpts {
                negated: instance.negated,
                case_insensitive: instance.case_insensitive,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let opts = pb::LikeOpts::decode(metadata)?;
        Ok(Some(LikeOptions {
            negated: opts.negated,
            case_insensitive: opts.case_insensitive,
        }))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 2 {
            vortex_bail!(
                "Like expression requires exactly 2 children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("child"),
            1 => ChildName::from("pattern"),
            _ => unreachable!("Invalid child index {} for Like expression", child_idx),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        expr.child(0).fmt_sql(f)?;
        if expr.data().negated {
            write!(f, " not")?;
        }
        if expr.data().case_insensitive {
            write!(f, " ilike ")?;
        } else {
            write!(f, " like ")?;
        }
        expr.child(1).fmt_sql(f)
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let input = expr.children()[0].return_dtype(scope)?;
        let pattern = expr.children()[1].return_dtype(scope)?;

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

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let child = expr.child(0).evaluate(scope)?;
        let pattern = expr.child(1).evaluate(scope)?;
        like_compute(&child, &pattern, *expr.data())
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
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_dtype::{DType, Nullability};

    use crate::Scope;
    use crate::exprs::get_item::get_item;
    use crate::exprs::like::{like, not_ilike};
    use crate::exprs::literal::lit;
    use crate::exprs::not::not;
    use crate::exprs::root::root;

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(&Scope::new(bools.to_array()))
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
