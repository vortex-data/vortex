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
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::arrays::VarBinView;
use crate::dtype::DType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
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

    /// Returns whether fill_null can execute inputs of `dtype`.
    ///
    /// These are the dtypes whose canonical arrays have a [`FillNullKernel`], so update this list
    /// whenever `fill_null_canonical` gains or loses one. Rewrites into fill_null, such as the
    /// CASE WHEN simplification, check it first because other dtypes fail at execution time.
    pub fn supports_dtype(dtype: &DType) -> bool {
        matches!(
            dtype,
            DType::Bool(_)
                | DType::Primitive(..)
                | DType::Decimal(..)
                | DType::Utf8(_)
                | DType::Binary(_)
        )
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
        vortex_ensure!(
            arg_dtypes[0].eq_ignore_nullability(&arg_dtypes[1]),
            "fill_null requires input and fill value to have the same base type, got {} and {}",
            arg_dtypes[0],
            arg_dtypes[1]
        );
        // The result dtype takes the nullability of the fill value.
        Ok(arg_dtypes[0]
            .clone()
            .with_nullability(arg_dtypes[1].nullability()))
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

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            if fill_was_lazy {
                // Hand the rebuilt node back with a constant fill so the input's FillNull
                // kernels get a chance before we decode the input.
                return FillNull::try_new(input, fill_value)?
                    .into_array()
                    .optimize(ctx.session());
            }
            let input = input.execute::<ArrayRef>(ctx)?;
            return Ok(FillNull::try_new(input, fill_value)?.into_array());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => fill_null_canonical(canonical, &fill_scalar, ctx),
            ColumnarView::Constant(constant) => fill_null_constant(constant, &fill_scalar),
        }
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

/// Fill nulls on a canonical array by directly dispatching to the appropriate kernel.
///
/// Returns the filled array, or bails if no kernel is registered for the canonical type.
fn fill_null_canonical(
    canonical: CanonicalView<'_>,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let arr = canonical.to_array_ref();
    if let Some(result) = short_circuit(&arr, fill_value)? {
        return Ok(result);
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
        other => vortex_bail!(
            "No FillNullKernel for canonical array {}",
            other.to_array_ref().encoding_id()
        ),
    }
}
