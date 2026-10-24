// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::ArrayView;
use crate::Canonical;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::ListView;
use crate::arrays::Map;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::arrays::VarBinView;
use crate::arrays::struct_::compute::cast::struct_cast;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::proto::expr as pb;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::display::ExprDisplay;
use crate::scalar_fn::is_not_null_node;

/// A cast expression that converts values to a target data type.
#[derive(Clone)]
pub struct Cast;

impl Cast {
    /// Creates a lazy cast of `input` to `target_dtype`.
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef, target_dtype: DType) -> ScalarFnArray {
        ScalarFnArray::try_new(Cast.bind(target_dtype), [input])
            .vortex_expect("Cast has one child and an infallible return dtype")
    }
}

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

    fn fmt_sql(
        &self,
        dtype: &DType,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "cast(")?;
        Display::fmt(expr.display_child(0), f)?;
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
                // Some array conversions, including bool to numeric, have no scalar cast.
                None => input
                    .execute::<Canonical>(ctx)?
                    .into_array()
                    .cast(target_dtype.clone()),
            },
        }
    }

    fn reduce<T: ReduceNode>(&self, target_dtype: &DType, node: &T) -> VortexResult<Option<T>> {
        // Collapse node if child is already the target type
        let child = node.child(0);
        if &child.node_dtype()? == target_dtype {
            return Ok(Some(child));
        }
        Ok(None)
    }

    fn simplify<T: ReduceNode>(&self, target_dtype: &DType, node: &T) -> VortexResult<Option<T>> {
        let child = node.child(0);
        if let Some(scalar) = child.as_constant() {
            // A failing cast (e.g. null to a non-nullable dtype) is left in place so the error
            // surfaces at execution time rather than during optimization.
            return Ok(scalar
                .cast(target_dtype)
                .ok()
                .map(|scalar| node.new_constant(scalar)));
        }
        // Collapse node if child is already the target type
        if &child.node_dtype()? == target_dtype {
            return Ok(Some(child));
        }
        Ok(None)
    }

    fn validity<T: ReduceNode>(
        &self,
        dtype: &DType,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        Ok(ReduceNodeValidity::Reduced(if dtype.is_nullable() {
            is_not_null_node(&node.child(0))?
        } else {
            node.new_constant(true.into())
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
        CanonicalView::Map(a) => <Map as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::FixedSizeList(a) => <FixedSizeList as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Struct(a) => struct_cast(a, dtype, ctx),
        CanonicalView::Union(_) => {
            vortex_bail!(
                "TODO(connor)[Union]: implement Union casting with conformance coverage for \
                 outer nullability changes, including validation of nullable-to-nonnullable casts"
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
