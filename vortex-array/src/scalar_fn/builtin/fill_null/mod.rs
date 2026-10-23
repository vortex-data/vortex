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
use crate::expr::BoundExpression;
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

    fn simplify(
        &self,
        _options: &Self::Options,
        expr: &BoundExpression,
    ) -> VortexResult<Option<BoundExpression>> {
        if !expr.child(0).dtype().is_nullable() {
            return Ok(Some(expr.child(0).clone()));
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::StructArray;
    use crate::arrays::scalar_fn::ExactScalarFn;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::MapDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::fill_null;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::kernel::ExecuteParentKernel;
    use crate::matcher::Matcher;
    use crate::optimizer::rules::ArrayParentReduceRule;
    use crate::scalar::Scalar;
    use crate::scalar_fn::builtin::binary::Binary;
    use crate::scalar_fn::builtin::operators::Operator;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// A fill of `40 + 2` that is not a [`ConstantArray`] until it executes.
    fn lazy_fill_42(len: usize) -> VortexResult<ArrayRef> {
        Ok(ScalarFnArray::try_new(
            Binary.bind(Operator::Add),
            vec![
                ConstantArray::new(40i32, len).into_array(),
                ConstantArray::new(2i32, len).into_array(),
            ],
        )?
        .into_array())
    }

    #[test]
    fn dtype() {
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        assert_eq!(
            fill_null(root(), lit(0i32)).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = fill_null(root(), lit(0i32));
        let replaced = expr
            .with_children(vec![get_item("a", root()), lit(42i32)])
            .vortex_expect("operation should succeed in test");
        assert_eq!(replaced, fill_null(get_item("a", root()), lit(42i32)));
    }

    #[test]
    fn evaluate() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)])
                .into_array();

        let expr = fill_null(root(), lit(42i32));
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([1i32, 42, 3, 42, 5]),
            &mut ctx
        );
    }

    #[test]
    fn evaluate_lazy_constant_fill() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let fill = lazy_fill_42(input.len())?;
        assert!(fill.as_constant().is_none());

        let result = FillNull::try_new(input, fill)?
            .into_array()
            .execute::<ArrayRef>(&mut ctx)?;

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 42, 3]), &mut ctx);
        Ok(())
    }

    #[test]
    fn evaluate_lazy_fill_preserves_chunked_input() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), None]).into_array(),
                PrimitiveArray::from_option_iter([Some(3), None]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::Nullable),
        )?
        .into_array();
        let fill = lazy_fill_42(input.len())?;

        let result = FillNull::try_new(input, fill)?
            .into_array()
            .execute::<ArrayRef>(&mut ctx)?;

        assert!(result.is::<Chunked>());
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([1i32, 42, 3, 42]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn non_constant_fill_declines_and_errors() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let varying = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

        let constant_input = ConstantArray::new(Scalar::null_native::<i32>(), 3).into_array();
        let constant_parent =
            FillNull::try_new(constant_input.clone(), varying.clone())?.into_array();
        let reduced = FillNullReduceAdaptor(Constant).reduce_parent(
            constant_input
                .as_opt::<Constant>()
                .vortex_expect("constant input"),
            ExactScalarFn::<FillNull>::try_match(&constant_parent)
                .vortex_expect("fill_null parent"),
            0,
        )?;
        assert!(reduced.is_none());

        let primitive_input =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let primitive_parent = FillNull::try_new(primitive_input.clone(), varying)?.into_array();
        let executed = FillNullExecuteAdaptor(Primitive).execute_parent(
            primitive_input
                .as_opt::<Primitive>()
                .vortex_expect("primitive input"),
            ExactScalarFn::<FillNull>::try_match(&primitive_parent)
                .vortex_expect("fill_null parent"),
            0,
            &mut ctx,
        )?;
        assert!(executed.is_none());

        let err = primitive_parent.execute::<ArrayRef>(&mut ctx).unwrap_err();
        assert!(
            err.to_string()
                .contains("fill_null fill_value must be a constant/scalar")
        );
        Ok(())
    }

    /// Filling a chunked array slices the constant fill per chunk, and an empty slice is a
    /// canonical array rather than a constant. Stacked fills over an empty chunk must still run.
    #[test]
    fn repeated_fill_over_empty_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter(Vec::<Option<u16>>::new()).into_array(),
                PrimitiveArray::from_option_iter([Some(7u16), None]).into_array(),
            ],
            DType::Primitive(PType::U16, Nullability::Nullable),
        )?
        .into_array();

        let result = input
            .fill_null(Scalar::primitive(1u16, Nullability::Nullable))?
            .fill_null(Scalar::primitive(2u16, Nullability::NonNullable))?;

        assert_arrays_eq!(result, PrimitiveArray::from_iter([7u16, 1]), &mut ctx);
        Ok(())
    }

    #[test]
    fn evaluate_struct_field() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
        )])
        .unwrap()
        .into_array();

        let expr = fill_null(get_item("a", root()), lit(0i32));
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 0, 3]), &mut ctx);
    }

    #[test]
    fn evaluate_non_nullable_input() {
        let mut ctx = SESSION.create_execution_ctx();
        let test_array = buffer![1i32, 2, 3].into_array();
        let expr = fill_null(root(), lit(0i32));
        let result = test_array.apply(&expr).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([1i32, 2, 3]), &mut ctx);
    }

    /// Inputs with both valid and null rows execute exactly for the dtypes that `supports_dtype`
    /// accepts, so adding a kernel without updating `supports_dtype` fails here.
    #[test]
    fn supports_dtype_matches_kernels() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let dtypes = [
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Decimal(DecimalDType::new(19, 2), Nullability::Nullable),
            DType::Utf8(Nullability::Nullable),
            DType::Binary(Nullability::Nullable),
            DType::List(Arc::new(i32_dtype.clone()), Nullability::Nullable),
            DType::FixedSizeList(Arc::new(i32_dtype.clone()), 2, Nullability::Nullable),
            DType::Struct(
                StructFields::from_iter([("value", i32_dtype.clone())]),
                Nullability::Nullable,
            ),
            DType::Map(
                MapDType::try_new(i32_dtype.clone(), i32_dtype, false)?,
                Nullability::Nullable,
            ),
            DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()),
        ];

        let supported = [
            true, true, true, true, true, false, false, false, false, false,
        ];
        for (dtype, supported) in dtypes.into_iter().zip(supported) {
            assert_eq!(FillNull::supports_dtype(&dtype), supported, "{dtype}");
            let fill = Scalar::default_value(&dtype.as_nonnullable());
            let mut builder = builder_with_capacity_in(&dtype, 2, ctx.allocator());
            builder.append_scalar(&fill.cast(&dtype)?)?;
            builder.append_null();
            let filled = builder
                .finish()
                .fill_null(fill.clone())?
                .execute::<Canonical>(&mut ctx);

            if supported {
                let filled = filled?.into_array();
                assert_eq!(filled.dtype(), fill.dtype(), "{dtype}");
                assert_eq!(filled.execute_scalar(1, &mut ctx)?, fill, "{dtype}");
                assert_arrays_eq!(filled, ConstantArray::new(fill, 2), &mut ctx);
            } else {
                assert!(filled.is_err(), "{dtype} executes but is not supported");
            }
        }
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = fill_null(get_item("value", root()), lit(0i32));
        assert_eq!(expr.to_string(), "vortex.fill_null($.value, 0i32)");
    }
}
