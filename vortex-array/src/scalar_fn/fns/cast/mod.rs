// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::ArrayView;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::ListView;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::VarBinView;
use crate::arrays::struct_::compute::cast::struct_cast;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::expression::Expression;
use crate::expr::lit;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::literal::Literal;

/// A cast expression that converts values to a target data type.
#[derive(Clone)]
pub struct Cast;

impl ScalarFnVTable for Cast {
    type Options = DType;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.cast");
        *ID
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

    fn execute(
        &self,
        target_dtype: &DType,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            return input.execute::<ArrayRef>(ctx)?.cast(target_dtype.clone());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => {
                match cast_canonical(canonical, target_dtype, ctx)? {
                    Some(result) => Ok(result),
                    None => vortex_bail!(
                        "No CastKernel to cast canonical array {} from {} to {}",
                        canonical.to_array_ref().encoding_id(),
                        canonical.to_array_ref().dtype(),
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

    fn simplify_untyped(
        &self,
        target_dtype: &DType,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let Some(scalar) = expr.child(0).as_opt::<Literal>() else {
            return Ok(None);
        };
        // A failing cast (e.g. null to a non-nullable dtype) is left in place so the error
        // surfaces at execution time rather than during optimization.
        Ok(scalar.cast(target_dtype).ok().map(lit))
    }

    fn validity(&self, dtype: &DType, expression: &Expression) -> VortexResult<Option<Expression>> {
        Ok(Some(if dtype.is_nullable() {
            expression.child(0).validity()?
        } else {
            lit(true)
        }))
    }

    fn is_strict(&self, _instance: &DType) -> bool {
        // Cast options can pin a non-nullable output dtype instead of propagating nullability.
        false
    }
}

/// Cast a canonical array to the target dtype by dispatching to the appropriate
/// [`CastKernel`] for each canonical encoding.
///
/// Canonical encodings that can manipulate validity directly all implement [`CastKernel`] —
/// the kernel is the execution-time complement of their [`CastReduce`] rule and can compute
/// statistics (e.g. min of the validity array) when the reduce rule had to give up.
/// Encodings that delegate to scalars or storage (e.g. [`Null`], [`Constant`], [`Extension`])
/// only implement [`CastReduce`] because they never need execution-level information.
fn cast_canonical(
    canonical: CanonicalView<'_>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    match canonical {
        CanonicalView::Null(a) => <Null as CastReduce>::cast(a, dtype),
        CanonicalView::Bool(a) => <Bool as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Primitive(a) => <Primitive as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Decimal(a) => <Decimal as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::VarBinView(a) => <VarBinView as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::List(a) => <ListView as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Map(_) => vortex_bail!("Map arrays don't support casting"),
        CanonicalView::FixedSizeList(a) => <FixedSizeList as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Struct(a) => struct_cast(a, dtype, ctx),
        CanonicalView::Union(_) => {
            todo!(
                "TODO(connor)[Union]: implement Union casting with conformance coverage for outer \
                 nullability changes, including validation of nullable-to-nonnullable casts"
            )
        }
        CanonicalView::Extension(a) => <Extension as CastReduce>::cast(a, dtype),
        CanonicalView::Variant(_) => {
            vortex_bail!("Variant arrays don't support casting")
        }
    }
}

/// Cast a constant array by dispatching to its [`CastReduce`] implementation.
fn cast_constant(array: ArrayView<Constant>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    <Constant as CastReduce>::cast(array, dtype)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::Cast;
    use crate::IntoArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::cast;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::literal::Literal;

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
    fn simplify_folds_cast_of_literal() -> VortexResult<()> {
        let expr = cast(
            lit(3i32),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype())?;

        let scalar = optimized
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
        assert_eq!(scalar, &Scalar::primitive(3.0f64, Nullability::NonNullable));
        Ok(())
    }

    #[test]
    fn simplify_folds_cast_of_decimal_literal() -> VortexResult<()> {
        let decimal = Scalar::decimal(
            DecimalValue::I128(319),
            DecimalDType::new(3, 2),
            Nullability::NonNullable,
        );
        let expr = cast(
            lit(decimal),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype())?;

        let scalar = optimized
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
        assert_eq!(
            scalar,
            &Scalar::primitive(3.19f64, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn simplify_leaves_failing_cast_unchanged() -> VortexResult<()> {
        let target = DType::Primitive(PType::F64, Nullability::NonNullable);
        let expr = cast(
            lit(Scalar::null(DType::Primitive(
                PType::I32,
                Nullability::Nullable,
            ))),
            target.clone(),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype())?;

        assert!(optimized.as_opt::<Literal>().is_none());
        assert_eq!(optimized.as_opt::<Cast>(), Some(&target));
        Ok(())
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
