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
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::arrays::Struct;
use crate::arrays::VarBinView;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::BoundExpression;
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
use crate::scalar_fn::fns::zip::Zip;
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
    /// These are the dtypes that `fill_null_canonical` can fill, so update this function whenever
    /// it gains or loses one. Rewrites into fill_null, such as the CASE WHEN simplification,
    /// check it first because other dtypes fail at execution time.
    ///
    /// A null dtype has no non-null fill value. Union and variant values have no zip, so any dtype
    /// that contains one of them is rejected, even where its fill does not zip those values.
    pub(crate) fn supports_dtype(dtype: &DType) -> bool {
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
                    .optimize_ctx(ctx);
            }
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
        CanonicalView::Struct(a) => <Struct as FillNullKernel>::fill_null(a, fill_value, ctx)?
            .ok_or_else(|| vortex_err!("FillNullKernel for StructArray returned None")),
        CanonicalView::Extension(a) => {
            <Extension as FillNullKernel>::fill_null(a, fill_value, ctx)?
                .ok_or_else(|| vortex_err!("FillNullKernel for ExtensionArray returned None"))
        }
        CanonicalView::List(_) | CanonicalView::FixedSizeList(_) | CanonicalView::Map(_) => {
            fill_null_with_zip(arr, fill_value)
        }
        CanonicalView::Null(_) | CanonicalView::Union(_) | CanonicalView::Variant(_) => {
            vortex_bail!("fill_null is not supported for {} arrays", arr.dtype())
        }
    }
}

/// Fills nulls by zipping the input with a constant fill over the input validity.
///
/// The zip kernels of the list encodings select whole lists, so this reuses them instead of
/// rebuilding offsets here. The zip result is nullable like the input, so a cast gives it the
/// nullability of the fill value. The result is lazy.
fn fill_null_with_zip(array: ArrayRef, fill_value: &Scalar) -> VortexResult<ArrayRef> {
    let len = array.len();
    let result_dtype = array
        .dtype()
        .with_nullability(fill_value.dtype().nullability());
    let fill = ConstantArray::new(fill_value.cast(&result_dtype)?, len).into_array();
    let is_valid = array.validity()?.to_array(len);

    Zip::try_new(array, fill, is_valid)?
        .into_array()
        .cast(result_dtype)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::StructArray;
    use crate::arrays::VariantArray;
    use crate::arrays::scalar_fn::ExactScalarFn;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::MapDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::fill_null;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::kernel::ExecuteParentKernel;
    use crate::matcher::Matcher;
    use crate::memory::BufferAllocatorRef;
    use crate::optimizer::rules::ArrayParentReduceRule;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::operators::Operator;

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

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    /// Inputs with both valid and null rows execute for every dtype that `supports_dtype`
    /// accepts, with a nullable or a non-nullable fill value.
    #[rstest]
    #[case::bool(DType::Bool(Nullability::Nullable))]
    #[case::primitive(DType::Primitive(PType::I32, Nullability::Nullable))]
    #[case::decimal(DType::Decimal(DecimalDType::new(19, 2), Nullability::Nullable))]
    #[case::utf8(DType::Utf8(Nullability::Nullable))]
    #[case::binary(DType::Binary(Nullability::Nullable))]
    #[case::list(DType::List(Arc::new(i32_dtype()), Nullability::Nullable))]
    #[case::fixed_size_list(DType::FixedSizeList(Arc::new(i32_dtype()), 2, Nullability::Nullable))]
    #[case::struct_(DType::struct_([("value", i32_dtype())], Nullability::Nullable))]
    #[case::map(DType::Map(map_dtype(), Nullability::Nullable,))]
    #[case::date(DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()))]
    fn supported_dtypes_fill_mixed_validity(
        #[case] dtype: DType,
        #[values(Nullability::NonNullable, Nullability::Nullable)] fill_nullability: Nullability,
    ) -> VortexResult<()> {
        assert!(FillNull::supports_dtype(&dtype), "{dtype}");

        let mut ctx = SESSION.create_execution_ctx();
        let fill = Scalar::default_value(&dtype.as_nonnullable())
            .cast(&dtype.with_nullability(fill_nullability))?;
        let mut builder = builder_with_capacity_in(&dtype, 2, ctx.allocator());
        builder.append_scalar(&fill.cast(&dtype)?)?;
        builder.append_null();
        let filled = builder
            .finish()
            .fill_null(fill.clone())?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert_eq!(filled.dtype(), fill.dtype(), "{dtype}");
        assert_eq!(filled.execute_scalar(1, &mut ctx)?, fill, "{dtype}");
        assert_arrays_eq!(filled, ConstantArray::new(fill, 2), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::null(DType::Null)]
    #[case::variant(DType::Variant(Nullability::Nullable))]
    #[case::nested_variant(DType::struct_(
        [("value", DType::Variant(Nullability::Nullable))],
        Nullability::Nullable,
    ))]
    fn unsupported_dtypes(#[case] dtype: DType) {
        assert!(!FillNull::supports_dtype(&dtype), "{dtype}");
    }

    /// A variant input with mixed validity fails with an error instead of a panic.
    #[test]
    fn variant_with_mixed_validity_errors() -> VortexResult<()> {
        let dtype = DType::Variant(Nullability::Nullable);
        let value = Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable));
        let core_storage = ChunkedArray::try_new(
            vec![
                ConstantArray::new(value.cast(&dtype)?, 1).into_array(),
                ConstantArray::new(Scalar::null(dtype.clone()), 1).into_array(),
            ],
            dtype,
        )?
        .into_array();
        let input = VariantArray::try_new(core_storage, None)?.into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let error = input
            .fill_null(value)?
            .execute::<Canonical>(&mut ctx)
            .unwrap_err();
        assert!(
            error.to_string().contains("fill_null is not supported"),
            "{error}"
        );
        Ok(())
    }

    fn array_from_scalars(dtype: &DType, scalars: &[Scalar]) -> VortexResult<ArrayRef> {
        let mut builder =
            builder_with_capacity_in(dtype, scalars.len(), BufferAllocatorRef::static_ref());
        for scalar in scalars {
            builder.append_scalar(scalar)?;
        }
        Ok(builder.finish())
    }

    fn nullable_i32(value: Option<i32>) -> Scalar {
        match value {
            Some(value) => Scalar::primitive(value, Nullability::Nullable),
            None => Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        }
    }

    fn nullable_i32_list(values: &[Option<i32>], nullability: Nullability) -> Scalar {
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            values.iter().copied().map(nullable_i32).collect(),
            nullability,
        )
    }

    fn point_dtype() -> DType {
        DType::struct_([("x", i32_dtype())], Nullability::NonNullable)
    }

    fn points(xs: &[i32], nullability: Nullability) -> Scalar {
        let points = xs
            .iter()
            .map(|&x| Scalar::struct_(point_dtype(), [Scalar::from(x)]))
            .collect();
        Scalar::fixed_size_list(Arc::new(point_dtype()), points, nullability)
    }

    fn map_dtype() -> MapDType {
        MapDType::try_new(i32_dtype(), DType::Utf8(Nullability::Nullable), false)
            .expect("valid map dtype")
    }

    fn map(entries: &[(i32, Option<&str>)], nullability: Nullability) -> Scalar {
        Scalar::map(
            DType::Map(map_dtype(), nullability),
            entries.iter().map(|&(key, value)| {
                let value = match value {
                    Some(value) => Scalar::utf8(value, Nullability::Nullable),
                    None => Scalar::null(DType::Utf8(Nullability::Nullable)),
                };
                (Scalar::from(key), value)
            }),
        )
    }

    fn items_dtype() -> DType {
        DType::List(Arc::new(i32_dtype()), Nullability::NonNullable)
    }

    fn items(values: &[i32], nullability: Nullability) -> Scalar {
        let dtype = DType::struct_([("items", items_dtype())], nullability);
        let items = Scalar::list(
            Arc::new(i32_dtype()),
            values.iter().copied().map(Scalar::from).collect(),
            Nullability::NonNullable,
        );
        Scalar::struct_(dtype, [items])
    }

    /// Null rows of nested dtypes take the whole fill value, while valid rows, including empty
    /// lists and null elements inside them, stay as they are. `None` input rows are null.
    #[rstest]
    #[case::list(
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::Nullable,
        ),
        |n| nullable_i32_list(&[Some(7), Some(8)], n),
        vec![
            Some(nullable_i32_list(&[Some(1), None], Nullability::Nullable)),
            None,
            Some(nullable_i32_list(&[], Nullability::Nullable)),
            None,
        ],
        |n| vec![
            nullable_i32_list(&[Some(1), None], n),
            nullable_i32_list(&[Some(7), Some(8)], n),
            nullable_i32_list(&[], n),
            nullable_i32_list(&[Some(7), Some(8)], n),
        ],
    )]
    #[case::fixed_size_list_of_structs(
        DType::FixedSizeList(Arc::new(point_dtype()), 2, Nullability::Nullable),
        |n| points(&[7, 8], n),
        vec![
            Some(points(&[1, 2], Nullability::Nullable)),
            None,
            Some(points(&[3, 4], Nullability::Nullable)),
        ],
        |n| vec![points(&[1, 2], n), points(&[7, 8], n), points(&[3, 4], n)],
    )]
    #[case::map(
        DType::Map(map_dtype(), Nullability::Nullable),
        |n| map(&[(9, Some("z")), (10, None)], n),
        vec![
            None,
            Some(map(&[(1, Some("a"))], Nullability::Nullable)),
            None,
            Some(map(&[], Nullability::Nullable)),
        ],
        |n| vec![
            map(&[(9, Some("z")), (10, None)], n),
            map(&[(1, Some("a"))], n),
            map(&[(9, Some("z")), (10, None)], n),
            map(&[], n),
        ],
    )]
    #[case::struct_of_lists(
        DType::struct_([("items", items_dtype())], Nullability::Nullable),
        |n| items(&[5, 6], n),
        vec![
            Some(items(&[1], Nullability::Nullable)),
            None,
            Some(items(&[], Nullability::Nullable)),
        ],
        |n| vec![items(&[1], n), items(&[5, 6], n), items(&[], n)],
    )]
    fn fill_null_nested(
        #[case] dtype: DType,
        #[case] fill: fn(Nullability) -> Scalar,
        #[case] input: Vec<Option<Scalar>>,
        #[case] expected: fn(Nullability) -> Vec<Scalar>,
        #[values(Nullability::NonNullable, Nullability::Nullable)] fill_nullability: Nullability,
    ) -> VortexResult<()> {
        let input = input
            .into_iter()
            .map(|row| row.unwrap_or_else(|| Scalar::null(dtype.clone())))
            .collect::<Vec<_>>();
        let input = array_from_scalars(&dtype, &input)?;
        let fill = fill(fill_nullability);
        let expected = array_from_scalars(fill.dtype(), &expected(fill_nullability))?;

        let mut ctx = SESSION.create_execution_ctx();
        let result = input.fill_null(fill)?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = fill_null(get_item("value", root()), lit(0i32));
        assert_eq!(expr.to_string(), "vortex.fill_null($.value, 0i32)");
    }

    /// A scalar function input is executed through the executor before its nulls are filled.
    #[test]
    fn executes_non_columnar_input_iteratively() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)])
            .into_array()
            .binary(ConstantArray::new(2i32, 3).into_array(), Operator::Gt)?;

        let result = input.fill_null(Scalar::from(false))?;

        assert_arrays_eq!(result, BoolArray::from_iter([false, false, true]), &mut ctx);
        Ok(())
    }
}
