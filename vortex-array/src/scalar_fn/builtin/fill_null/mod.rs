// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::Canonical;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::arrays::VarBinView;
use crate::dtype::DType;
use crate::optimizer::ArrayOptimizer;
use crate::require_input;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::is_not_null_node;

/// An expression that replaces null values in the input with a fill value.
#[derive(Clone)]
pub struct FillNull;

impl FillNull {
    /// Creates a lazy operation that replaces null input values with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or incompatible dtypes.
    pub fn try_new(input: ArrayRef, fill_value: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(FillNull.bind(EmptyOptions), [input, fill_value])
    }

    pub(crate) fn output_dtype(input: &DType, fill_value: &DType) -> VortexResult<DType> {
        vortex_ensure!(
            input.eq_ignore_nullability(fill_value),
            "fill_null requires input and fill value to have the same base type, got {input} and {fill_value}"
        );

        Ok(input.with_nullability(fill_value.nullability()))
    }

    /// Returns whether fill_null can execute inputs of `dtype`.
    ///
    /// These are the dtypes that `fill_null_canonical` and the `vortex-functions` fill_null
    /// kernels can fill, so update this function whenever they gain or lose one. Rewrites into
    /// fill_null, such as the CASE WHEN simplification, check it first because other dtypes fail
    /// at execution time.
    ///
    /// A null dtype has no non-null fill value. Union and variant values have no zip, so any dtype
    /// that contains one of them is rejected, even where its fill does not zip those values.
    pub fn supports_dtype(dtype: &DType) -> bool {
        !matches!(dtype, DType::Null) && !contains_union_or_variant(dtype)
    }
}

impl ScalarFnVTable for FillNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.fill_null");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            1 => ChildName::from("fill_value"),
            _ => unreachable!("Invalid child index {} for FillNull expression", child_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        Self::output_dtype(&arg_dtypes[0], &arg_dtypes[1])
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let fill_value = args.get(1)?;
        // Parent kernels decline a fill that is not already constant. Materialize it here so a
        // lazy scalar, such as a literal `ScalarFnArray`, can still fill nulls.
        let fill_was_lazy = !fill_value.is::<Constant>();
        let fill_value = if fill_was_lazy {
            fill_value.execute_until::<Constant>(ctx)?
        } else {
            fill_value
        };

        let Some(fill_scalar) = fill_value.as_constant() else {
            // Slicing a constant fill to an empty range yields an empty canonical array rather
            // than a constant. An empty input has no nulls to fill, so there is nothing to do.
            vortex_ensure!(
                input.is_empty(),
                "fill_null fill_value must be a constant/scalar"
            );
            let dtype = input
                .dtype()
                .with_nullability(fill_value.dtype().nullability());
            return Ok(Canonical::empty(&dtype).into_array());
        };

        vortex_ensure!(
            !fill_scalar.is_null(),
            "fill_null requires a non-null fill value"
        );

        if fill_was_lazy {
            // Hand the rebuilt node back with a constant fill so the input's FillNull
            // kernels (including those registered by vortex-functions) get a chance.
            return FillNull::try_new(input, fill_value)?
                .into_array()
                .optimize_ctx(ctx);
        }

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            let input = input.execute::<ArrayRef>(ctx)?;
            return Ok(FillNull::try_new(input, fill_value)?.into_array());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => fill_null_canonical(canonical, fill_scalar, ctx),
            ColumnarView::Constant(constant) => fill_null_constant(constant, fill_scalar),
        }
    }

    fn execute_step(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        // Only once the fill is constant: `execute` materializes a lazy fill first, so the input's
        // FillNull kernels can still apply before the input is decoded.
        if args.input_matches(1, |fill| fill.is::<Constant>())? {
            require_input!(args, 0 => AnyColumnar);
        }
        self.execute(options, args, ctx)
            .map(ScalarFnExecutionResult::done)
    }

    fn simplify<T: ReduceNode>(
        &self,
        _options: &Self::Options,
        node: &T,
    ) -> VortexResult<Option<T>> {
        let child = node.child(0);
        if !child.node_dtype()?.is_nullable() {
            return Ok(Some(child));
        }
        Ok(None)
    }

    fn validity<T: ReduceNode>(
        &self,
        _options: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        // After fill_null, the result validity depends on the fill value's nullability.
        // If fill_value is non-nullable, the result is always valid.
        Ok(ReduceNodeValidity::Reduced(is_not_null_node(
            &node.child(1),
        )?))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        // This function replaces null input values instead of propagating them.
        false
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Returns whether `dtype` is or contains a union or a variant.
fn contains_union_or_variant(dtype: &DType) -> bool {
    match dtype {
        DType::Union(..) | DType::Variant(_) => true,
        DType::List(element, _) | DType::FixedSizeList(element, ..) => {
            contains_union_or_variant(element)
        }
        DType::Map(map, _) => {
            contains_union_or_variant(&map.key_dtype())
                || contains_union_or_variant(&map.value_dtype())
        }
        DType::Struct(fields, _) => fields
            .fields()
            .any(|field| contains_union_or_variant(&field)),
        DType::Extension(ext) => contains_union_or_variant(ext.storage_dtype()),
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => false,
    }
}

/// Fill nulls on a canonical array by directly dispatching to the appropriate kernel.
///
/// Returns the filled array, or bails if the canonical type cannot be filled.
fn fill_null_canonical(
    canonical: CanonicalView<'_>,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let arr = canonical.to_array_ref();
    if let Some(result) = short_circuit(&arr, fill_value)? {
        return Ok(result);
    }
    if !FillNull::supports_dtype(arr.dtype()) {
        vortex_bail!("fill_null is not supported for {} arrays", arr.dtype());
    }
    match canonical {
        CanonicalView::Bool(a) => <Bool as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for BoolArray returned None")),
        CanonicalView::Primitive(a) => {
            <Primitive as FillNullKernel>::fill_null(a, fill_value, ctx)?
                .ok_or_else(|| vortex_err!("FillNullKernel for PrimitiveArray returned None"))
        }
        CanonicalView::Decimal(a) => <Decimal as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for DecimalArray returned None")),
        CanonicalView::VarBinView(a) => {
            <VarBinView as FillNullKernel>::fill_null(a, fill_value, ctx)?
                .ok_or_else(|| vortex_err!("FillNullKernel for VarBinViewArray returned None"))
        }
        CanonicalView::Extension(a) => {
            <Extension as FillNullKernel>::fill_null(a, fill_value, ctx)?
                .ok_or_else(|| vortex_err!("FillNullKernel for ExtensionArray returned None"))
        }
        CanonicalView::Struct(_)
        | CanonicalView::List(_)
        | CanonicalView::FixedSizeList(_)
        | CanonicalView::Map(_) => {
            vortex_bail!(
                "fill_null for {} arrays requires vortex_functions::initialize",
                arr.dtype()
            )
        }
        CanonicalView::Null(_) | CanonicalView::Union(_) | CanonicalView::Variant(_) => {
            vortex_bail!("fill_null is not supported for {} arrays", arr.dtype())
        }
    }
}
