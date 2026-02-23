// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::arrays::BoolVTable;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionVTable;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewVTable;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructVTable;
use crate::arrays::VarBinViewVTable;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::ReduceCtx;
use crate::expr::ReduceNode;
use crate::expr::ReduceNodeRef;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::expression::Expression;
use crate::expr::lit;
use crate::expr::stats::Stat;

/// A cast expression that converts values to a target data type.
pub struct Cast;

impl VTable for Cast {
    type Options = DType;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.cast")
    }

    fn serialize(&self, dtype: &DType) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CastOpts {
                target: Some(dtype.try_into()?),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let proto = pb::CastOpts::decode(_metadata)?.target;
        DType::from_proto(
            proto
                .as_ref()
                .ok_or_else(|| vortex_err!("Missing target dtype in Cast expression"))?,
            session,
        )
    }

    fn arity(&self, _options: &DType) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &DType, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Cast expression", child_idx),
        }
    }

    fn fmt_sql(&self, dtype: &DType, expr: &Expression, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "cast(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, " as {}", dtype)?;
        write!(f, ")")
    }

    fn return_dtype(&self, dtype: &DType, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(dtype.clone())
    }

    fn execute(&self, target_dtype: &DType, mut args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let input = args
            .inputs
            .pop()
            .vortex_expect("missing input for Cast expression");

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            return input
                .execute::<ArrayRef>(args.ctx)?
                .cast(target_dtype.clone());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => {
                match cast_canonical(canonical.clone(), target_dtype, args.ctx)? {
                    Some(result) => Ok(result),
                    None => vortex_bail!(
                        "No CastKernel to cast canonical array {} from {} to {}",
                        canonical.as_ref().encoding_id(),
                        canonical.as_ref().dtype(),
                        target_dtype,
                    ),
                }
            }
            ColumnarView::Constant(constant) => match cast_constant(constant, target_dtype)? {
                Some(result) => Ok(result),
                None => vortex_bail!(
                    "No CastReduce to cast constant array from {} to {}",
                    constant.dtype(),
                    target_dtype,
                ),
            },
        }
    }

    fn reduce(
        &self,
        target_dtype: &DType,
        node: &dyn ReduceNode,
        _ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        // Collapse node if child is already the target type
        let child = node.child(0);
        if &child.node_dtype()? == target_dtype {
            return Ok(Some(child));
        }
        Ok(None)
    }

    fn stat_expression(
        &self,
        dtype: &DType,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        match stat {
            Stat::IsConstant
            | Stat::IsSorted
            | Stat::IsStrictSorted
            | Stat::NaNCount
            | Stat::Sum
            | Stat::UncompressedSizeInBytes => expr.child(0).stat_expression(stat, catalog),
            Stat::Max | Stat::Min => {
                // We cast min/max to the new type
                expr.child(0)
                    .stat_expression(stat, catalog)
                    .map(|x| cast(x, dtype.clone()))
            }
            Stat::NullCount => {
                // if !expr.data().is_nullable() {
                // NOTE(ngates): we should decide on the semantics here. In theory, the null
                //  count of something cast to non-nullable will be zero. But if we return
                //  that we know this to be zero, then a pruning predicate may eliminate data
                //  that would otherwise have caused the cast to error.
                // return Some(lit(0u64));
                // }
                None
            }
        }
    }

    fn validity(&self, dtype: &DType, expression: &Expression) -> VortexResult<Option<Expression>> {
        Ok(Some(if dtype.is_nullable() {
            expression.child(0).validity()?
        } else {
            lit(true)
        }))
    }

    // This might apply a nullability
    fn is_null_sensitive(&self, _instance: &DType) -> bool {
        true
    }
}

/// Cast a canonical array to the target dtype by dispatching to the appropriate
/// [`CastReduce`] or [`CastKernel`] for each canonical encoding.
fn cast_canonical(
    canonical: CanonicalView<'_>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    match canonical {
        CanonicalView::Null(a) => <NullVTable as CastReduce>::cast(a, dtype),
        CanonicalView::Bool(a) => <BoolVTable as CastReduce>::cast(a, dtype),
        CanonicalView::Primitive(a) => <PrimitiveVTable as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Decimal(a) => <DecimalVTable as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::VarBinView(a) => <VarBinViewVTable as CastReduce>::cast(a, dtype),
        CanonicalView::List(a) => <ListViewVTable as CastReduce>::cast(a, dtype),
        CanonicalView::FixedSizeList(a) => <FixedSizeListVTable as CastReduce>::cast(a, dtype),
        CanonicalView::Struct(a) => <StructVTable as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Extension(a) => <ExtensionVTable as CastReduce>::cast(a, dtype),
    }
}

/// Cast a constant array by dispatching to its [`CastReduce`] implementation.
fn cast_constant(array: &ConstantArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    <ConstantVTable as CastReduce>::cast(array, dtype)
}

/// Creates an expression that casts values to a target data type.
///
/// Converts the input expression's values to the specified target type.
///
/// ```rust
/// # use vortex_array::dtype::{DType, Nullability, PType};
/// # use vortex_array::expr::{cast, root};
/// let expr = cast(root(), DType::Primitive(PType::I64, Nullability::NonNullable));
/// ```
pub fn cast(child: Expression, target: DType) -> Expression {
    Cast.try_new_expr(target, [child])
        .vortex_expect("Failed to create Cast expression")
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;

    use super::cast;
    use crate::IntoArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::root::root;
    use crate::expr::test_harness;

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
        expr.with_children(vec![root()])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: Expression = cast(
            get_item("a", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = test_array.apply(&expr).unwrap();

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
        assert_eq!(expr.to_string(), "cast($.value as i64)");

        let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
        assert_eq!(expr2.to_string(), "cast($ as bool?)");
    }
}
