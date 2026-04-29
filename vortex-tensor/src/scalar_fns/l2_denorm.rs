// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 denormalization expression for normalized vectors.

use std::fmt::Formatter;

use num_traits::Float;
use num_traits::ToPrimitive;
use num_traits::Zero;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn as ScalarFnArrayEncoding;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::scalar_fns::l2_norm::L2Norm;
use crate::types::normalized_vector::AnyNormalizedVector;
use crate::types::normalized_vector::NormalizedVector;
use crate::types::vector::AnyVector;
use crate::types::vector::Vector;
use crate::utils::extract_constant_flat_row;
use crate::utils::extract_flat_elements;
use crate::utils::extract_l2_denorm_children;
use crate::utils::inner_vector_array;
use crate::utils::unit_norm_tolerance;
use crate::utils::vector_fsl_storage_dtype;

/// Re-applies authoritative L2 norms to a normalized vector column.
///
/// Computes `normalized * norm` on each row over the flat backing buffer of the vector child.
///
/// The first child must be a [`NormalizedVector`]. Exact callers should use
/// [`try_new_array`](Self::try_new_array), which verifies that stored norms are non-negative and
/// that a zero stored norm is paired with an all-zero normalized row. Lossy encodings may use
/// [`new_array_unchecked`](Self::new_array_unchecked) when the decoded child is only an
/// approximation but the stored norms are still authoritative.
///
/// The norms input must be a primitive float column with the same element type as the
/// normalized vector elements.
///
/// Downstream readthrough rules intentionally treat the stored norms and normalized child as the
/// encoding contract, even when that differs slightly from recomputing over fully decoded
/// coordinates.
///
/// [`NormalizedVector`]: crate::normalized_vector::NormalizedVector
#[derive(Clone)]
pub struct L2Denorm;

impl L2Denorm {
    /// Creates a new [`TypedScalarFnInstance`] wrapping the L2 denormalization operation.
    ///
    /// This is a low-level scalar-function descriptor constructor. To build a semantically valid
    /// [`L2Denorm`] array, prefer [`try_new_array`](Self::try_new_array).
    pub fn new() -> TypedScalarFnInstance<L2Denorm> {
        TypedScalarFnInstance::new(L2Denorm, EmptyOptions)
    }

    /// Constructs a validated [`ScalarFnArray`] that lazily re-applies `norms` to `normalized`.
    ///
    /// In addition to the structural checks performed by [`ScalarFnArray::try_new`], this
    /// constructor verifies that the first child is a [`NormalizedVector`], that stored norms are
    /// non-negative, and that any row with stored norm `0.0` has an all-zero normalized row.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches), if a stored norm is negative, or if a zero-norm row is paired with a
    /// non-zero normalized row.
    pub fn try_new_array(
        normalized: ArrayRef,
        norms: ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnArray> {
        validate_norms_against_normalized(&normalized, &norms, ctx)?;

        // SAFETY: The validation above established the exact L2Denorm invariants.
        unsafe { Self::new_array_unchecked(normalized, norms, len) }
    }

    /// Constructs an [`L2Denorm`] array without validating row values against `norms`.
    ///
    /// Structural validation still runs via [`ScalarFnArray::try_new`], so the first child must be
    /// a [`NormalizedVector`]. Use this when the normalized child is a lossy approximation whose
    /// rows may not be exactly unit-norm or may not preserve exact zero-ness.
    ///
    /// # Safety
    ///
    /// The caller must ensure the first child is semantically suitable for L2 denormalization and
    /// is wrapped as a [`NormalizedVector`]. For exact wrappers, every valid row must be unit-norm
    /// or zero and stored norms must be non-negative. Lossy encodings may deliberately relax the
    /// exact row invariant while still treating the stored norms as authoritative.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub unsafe fn new_array_unchecked(
        normalized: ArrayRef,
        norms: ArrayRef,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(L2Denorm::new().erased(), vec![normalized, norms], len)
    }
}

impl ScalarFnVTable for L2Denorm {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.tensor.l2_denorm")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("normalized"),
            1 => ChildName::from("norms"),
            _ => unreachable!("L2Denorm must have exactly two children"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "l2_denorm(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let normalized = &arg_dtypes[0];
        let norms = &arg_dtypes[1];

        l2_denorm_output_dtype(normalized, norms)
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let normalized_ref = args.get(0)?;
        let norms_ref = args.get(1)?;
        let output_dtype = l2_denorm_output_dtype(normalized_ref.dtype(), norms_ref.dtype())?;
        let validity = normalized_ref.validity()?.and(norms_ref.validity()?)?;

        if let Some(const_norms) = norms_ref.as_opt::<Constant>() {
            let norm_scalar = const_norms.scalar();
            vortex_ensure!(
                norm_scalar.dtype().is_float(),
                "L2Denorm constant norms must be a float scalar, got {}",
                norm_scalar.dtype(),
            );

            if let Some(norm_value) = norm_scalar.value() {
                return execute_l2_denorm_constant_norms(
                    normalized_ref,
                    norm_scalar,
                    norm_value,
                    output_dtype,
                    validity,
                    ctx,
                );
            }
        }

        // Drill past any `NormalizedVector` wrapper so we always work with the underlying
        // `Vector` extension array.
        let vector_ref = inner_vector_array(&normalized_ref, ctx)?;
        let normalized: ExtensionArray = vector_ref.execute(ctx)?;
        let norms: PrimitiveArray = norms_ref.execute(ctx)?;
        let row_count = args.row_count();

        let vector_metadata = normalized
            .dtype()
            .as_extension()
            .metadata_opt::<AnyVector>()
            .vortex_expect("we already validated this in `return_dtype`");
        let tensor_flat_size = vector_metadata.dimensions() as usize;

        let flat = extract_flat_elements(normalized.storage_array(), tensor_flat_size, ctx)?;

        // TODO(connor): Do we want a "broadcast" expression for the List types, or is this fine?
        match_each_float_ptype!(flat.ptype(), |T| {
            let norms = norms.as_slice::<T>();

            let elements: Buffer<T> = (0..row_count)
                .flat_map(|i| {
                    let norm = norms[i];
                    flat.row::<T>(i).iter().map(move |&x| x * norm)
                })
                .collect();

            build_tensor_array(
                output_dtype,
                tensor_flat_size,
                row_count,
                validity,
                elements,
            )
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let normalized_validity = expression.child(0).validity()?;
        let norms_validity = expression.child(1).validity()?;

        Ok(Some(and(normalized_validity, norms_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Returns the denormalized output dtype for a normalized vector child and matching norms.
fn l2_denorm_output_dtype(normalized: &DType, norms: &DType) -> VortexResult<DType> {
    let normalized_ext = normalized.as_extension_opt().ok_or_else(|| {
        vortex_err!("L2Denorm normalized child must be a NormalizedVector, got {normalized}")
    })?;
    let normalized_metadata = normalized_ext
        .metadata_opt::<AnyNormalizedVector>()
        .ok_or_else(|| {
            vortex_err!("L2Denorm normalized child must be a NormalizedVector, got {normalized}")
        })?;
    let element_ptype = normalized_metadata.element_ptype();

    let DType::Primitive(norms_ptype, _) = norms else {
        vortex_bail!("L2Denorm norms must be a primitive float array, got {norms}");
    };
    vortex_ensure!(
        norms_ptype.is_float(),
        "L2Denorm norms must be a primitive float array, got {norms}",
    );
    vortex_ensure_eq!(
        *norms_ptype,
        element_ptype,
        "L2Denorm norms dtype must match normalized element dtype ({element_ptype}), \
            got {norms_ptype}",
    );

    let fsl_dtype = vector_fsl_storage_dtype(normalized_ext).ok_or_else(|| {
        vortex_err!("L2Denorm normalized child must be a NormalizedVector, got {normalized}")
    })?;
    let output = DType::Extension(ExtDType::<Vector>::try_new(EmptyMetadata, fsl_dtype)?.erased());

    Ok(output.union_nullability(norms.nullability()))
}

/// Metadata for a serialized [`L2Denorm`] array: both children's full [`DType`]s. The parent's
/// dtype is `normalized.union_nullability(norms.nullability())`, which loses both children's
/// individual nullabilities, so we persist them directly.
#[derive(Clone, prost::Message)]
pub(super) struct L2DenormMetadata {
    #[prost(message, optional, tag = "1")]
    normalized_dtype: Option<pb::DType>,
    #[prost(message, optional, tag = "2")]
    norms_dtype: Option<pb::DType>,
}

impl ScalarFnArrayVTable for L2Denorm {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let scalar_fn_array = view.as_::<ScalarFnArrayEncoding>();
        let normalized_dtype = Some(scalar_fn_array.child_at(0).dtype().try_into()?);
        let norms_dtype = Some(scalar_fn_array.child_at(1).dtype().try_into()?);
        Ok(Some(
            L2DenormMetadata {
                normalized_dtype,
                norms_dtype,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let metadata = L2DenormMetadata::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode L2DenormMetadata: {e}"))?;
        let normalized_pb = metadata
            .normalized_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2DenormMetadata missing normalized_dtype"))?;
        let norms_pb = metadata
            .norms_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2DenormMetadata missing norms_dtype"))?;
        let normalized_dtype = DType::from_proto(normalized_pb, session)?;
        let norms_dtype = DType::from_proto(norms_pb, session)?;
        let normalized = children.get(0, &normalized_dtype, len)?;
        let norms = children.get(1, &norms_dtype, len)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![normalized, norms],
        })
    }
}

/// Optimized execution when the norms array is constant.
fn execute_l2_denorm_constant_norms(
    normalized_ref: ArrayRef,
    norm_scalar: &Scalar,
    norm_value: &ScalarValue,
    output_dtype: DType,
    new_validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // If the norms are all equal to 1 then we don't need to do anything.
    let err = norm_value
        .as_primitive()
        .as_f64()
        .vortex_expect("we know that this is a float, so it must fit in f64")
        - 1.0f64;

    let normalized_metadata = normalized_ref
        .dtype()
        .as_extension_opt()
        .and_then(|ext| ext.metadata_opt::<AnyNormalizedVector>())
        .ok_or_else(|| {
            vortex_err!(
                "L2Denorm normalized child must be a NormalizedVector, got {}",
                normalized_ref.dtype(),
            )
        })?;

    let tolerance = unit_norm_tolerance(
        norm_scalar.dtype().as_ptype(),
        normalized_metadata.dimensions() as usize,
    );

    // Drill past any outer `NormalizedVector` wrapper so we always work with the inner plain
    // `Vector` extension array (and its `FixedSizeList` storage).
    let vector_ref = inner_vector_array(&normalized_ref, ctx)?;

    if err.abs() < tolerance {
        // The output dtype is the sibling plain `Vector`. Rebuild the FSL wrapper with the
        // combined validity so the executed array's storage nullability matches `output_dtype`.
        let normalized: ExtensionArray = vector_ref.execute(ctx)?;

        let storage_fsl: FixedSizeListArray = normalized.storage_array().clone().execute(ctx)?;
        let new_fsl = FixedSizeListArray::try_new(
            storage_fsl.elements().clone(),
            storage_fsl.list_size(),
            new_validity,
            storage_fsl.len(),
        )?;

        return Ok(ExtensionArray::try_new(
            output_dtype.as_extension().clone(),
            new_fsl.into_array(),
        )?
        .into_array());
    }

    // Even if the norms are not all 1, if they are all the same then we can multiply
    // the entire elements array by the same number.
    let normalized: ExtensionArray = vector_ref.execute(ctx)?;
    let storage_fsl: FixedSizeListArray = normalized.storage_array().clone().execute(ctx)?;

    // Replace the elements array with an array that multiplies it by the constant
    // norms array (with length multiplied by the dimensions of the vectors).
    let const_array =
        ConstantArray::new(norm_scalar.clone(), storage_fsl.elements().len()).into_array();
    let mult_elements = storage_fsl
        .elements()
        .clone()
        .binary(const_array, Operator::Mul)?;

    // SAFETY: We just updated the elements of the array with a scalar fn, so all
    // invariants still hold.
    let new_fsl = unsafe {
        FixedSizeListArray::new_unchecked(
            mult_elements,
            storage_fsl.list_size(),
            new_validity,
            storage_fsl.len(),
        )
    };

    Ok(ExtensionArray::new(output_dtype.as_extension().clone(), new_fsl.into_array()).into_array())
}

/// Builds an unexecuted [`L2Denorm`] expression by normalizing a vector input and reattaching the
/// exact norms as the `norms` child.
///
/// The returned array is a lazy `L2Denorm(normalized, norms)` scalar function array.
///
/// # Normalized child
///
/// For plain [`Vector`] input, every non-null row with a positive L2 norm is divided by its norm
/// to produce a unit-norm vector, and the normalized child is promoted to [`NormalizedVector`].
/// The normalized child is forced **non-nullable** with [`Validity::NonNullable`] so optimized
/// kernels only have to reason about unit-norm vs. zero rows, not nulls. Rows that are null in the
/// original input are **zeroed out** in the normalized output to avoid leaking undefined physical
/// storage values into downstream encodings.
///
/// For [`NormalizedVector`] input, the function takes a fast path that returns the input
/// unchanged as the normalized child and asks [`L2Norm`] for the per-row norms. The fast path
/// preserves the input's outer nullability rather than rewriting null rows to zero, since the
/// caller has already committed to a [`NormalizedVector`] shape and we do not want to
/// re-allocate storage just to coerce nullability.
///
/// # Nullability
///
/// Nullability is tracked entirely by the `norms` child. Null input rows produce null `norms` via
/// [`L2Norm`]'s validity propagation. When the [`L2Denorm`] wrapper is executed, the output
/// validity is `and(normalized_validity, norms_validity)`, which correctly identifies
/// originally-null rows.
///
/// Because this helper computes exact `norms` and (on the slow path) divides by them, the
/// returned `normalized` child satisfies the unit-norm invariant required by [`L2Denorm`].
///
/// [`NormalizedVector`]: crate::normalized_vector::NormalizedVector
pub fn normalize_as_l2_denorm(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnArray> {
    let row_count = input.len();
    let input_dtype = input.dtype().clone();
    let vector_metadata = input_dtype
        .as_extension_opt()
        .and_then(|ext| ext.metadata_opt::<AnyVector>())
        .ok_or_else(|| {
            vortex_err!(
                "normalize_as_l2_denorm requires a Vector or NormalizedVector extension input, \
                 got {input_dtype}",
            )
        })?;
    let tensor_flat_size = vector_metadata.dimensions() as usize;

    // Fast path: input is already a `NormalizedVector`. The slow path below would compute exact
    // norms and divide every row by its norm, but for a `NormalizedVector` the divisor is always
    // 1.0 (or 0.0 for zero rows). Skip the divide entirely and reattach `L2Norm`'s
    // short-circuited per-row 0.0 / 1.0 norms. Crucially, this preserves the invariant required
    // by [`L2Denorm::try_new_array`] that a zero-norm row is paired with an all-zero normalized
    // row, because [`L2Norm`]'s `NormalizedVector` short-circuit emits 0.0 exactly when the row
    // is all zero.
    // This also has the added benefit of correcting any lossy-encoded `NormalizedVector` arrays.
    if vector_metadata.is_normalized() {
        let norms_sfn = L2Norm::try_new_array(input.clone(), row_count)?;
        let norms_array: ArrayRef = norms_sfn.into_array().execute(ctx)?;

        // SAFETY: `input` is a `NormalizedVector`, so every valid row is unit-norm or zero by
        // type. `norms_array` was produced by `L2Norm`, so stored norms are non-negative and a
        // zero norm is always paired with an all-zero row.
        return unsafe { L2Denorm::new_array_unchecked(input, norms_array, row_count) };
    }

    // Constant fast path: if the input is a constant-backed extension, normalize the single
    // stored row once and return an `L2Denorm` whose children are both `ConstantArray`s.
    if let Some(wrapped) = try_build_constant_l2_denorm_from_constant(&input, row_count, ctx)? {
        return Ok(wrapped);
    }

    // Calculate the norms of the vectors.
    let norms_sfn = L2Norm::try_new_array(input.clone(), row_count)?;
    let norms_array: ArrayRef = norms_sfn.into_array().execute(ctx)?;
    let primitive_norms: PrimitiveArray = norms_array.clone().execute(ctx)?;
    let norms_validity = primitive_norms.validity()?;

    let input: ExtensionArray = input.execute(ctx)?;
    let flat = extract_flat_elements(input.storage_array(), tensor_flat_size, ctx)?;

    // Normalize all of the vectors.
    let normalized_storage = match_each_float_ptype!(flat.ptype(), |T| {
        let norm_values = primitive_norms.as_slice::<T>();

        let total_elements = row_count * tensor_flat_size;
        let mut elements = BufferMut::<T>::with_capacity(total_elements);
        for i in 0..row_count {
            let is_valid = norms_validity.is_valid(i)?;
            let norm = norm_values[i];

            // SAFETY: We allocated `row_count * tensor_flat_size` capacity and push exactly
            // `tensor_flat_size` elements per row.

            // Null rows must be explicitly zeroed out.
            if !is_valid || norm == T::zero() {
                unsafe { elements.push_n_unchecked(T::zero(), tensor_flat_size) };
            } else {
                for &x in flat.row::<T>(i) {
                    unsafe { elements.push_unchecked(x / norm) };
                }
            }
        }

        // Since L2Denorm's validity is the `and` of its child validities, we can make the
        // normalized child non-nullable.
        build_normalized_storage(tensor_flat_size, row_count, elements.freeze())
    })?;

    // SAFETY:
    // - `norms_array` was produced by `L2Norm(input)`, so every stored norm is non-negative and
    //   null rows already carry null validity through that child.
    // - For every valid row, we either emit all zeros when the norm is zero or divide every
    //   element by the exact stored norm, so the normalized storage is unit-norm (or zero) by
    //   construction.
    // - Null rows are zeroed out above to avoid propagating arbitrary physical storage values
    //   into downstream lossy encodings.
    let normalized = unsafe { NormalizedVector::new_unchecked(normalized_storage) }?;
    unsafe { L2Denorm::new_array_unchecked(normalized, norms_array, row_count) }
}

/// Attempts to build an [`L2Denorm`] whose two children are both [`ConstantArray`]s by eagerly
/// normalizing `input`'s single stored row.
///
/// Returns `Ok(None)` when `input` is not a plain vector extension array whose storage is a
/// [`ConstantArray`] with a non-null fixed-size-list scalar, or when it is already a
/// [`NormalizedVector`].
///
/// When `input` matches, the returned [`ScalarFnArray`] is equivalent to [`normalize_as_l2_denorm`]
/// but runs in `O(list_size)` time instead of `O(row_count * list_size)`.
///
/// This is helpful in some of the reduction steps for cosine similarity execution into inner
/// product execution.
pub(crate) fn try_build_constant_l2_denorm_from_constant(
    input: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ScalarFnArray>> {
    let Some(ext) = input.as_opt::<Extension>() else {
        return Ok(None);
    };
    let storage = ext.storage_array();
    let Some(const_storage) = storage.as_opt::<Constant>() else {
        return Ok(None);
    };
    if const_storage.scalar().is_null() {
        return Ok(None);
    }

    let Some(vector_metadata) = input
        .dtype()
        .as_extension_opt()
        .and_then(|ext| ext.metadata_opt::<AnyVector>())
    else {
        return Ok(None);
    };
    if vector_metadata.is_normalized() {
        return Ok(None);
    }

    let list_size = vector_metadata.dimensions() as usize;
    let original_nullability = input.dtype().nullability();
    let storage_fsl_nullability = storage.dtype().nullability();

    // Materialize just the single stored row; this does not expand the constant to the full
    // column length.
    let flat = extract_constant_flat_row(storage, ctx)?;

    let (normalized_fsl_scalar, norms_scalar) = match_each_float_ptype!(flat.ptype(), |T| {
        let row = flat.as_slice::<T>();

        let mut sum_sq = T::zero();
        for &x in row {
            sum_sq += x * x;
        }
        let norm_t: T = sum_sq.sqrt();

        // Zero-norm rows must be stored as all-zeros so the `NormalizedVector` invariant holds.
        // This mirrors the per-row logic in `normalize_as_l2_denorm`.
        let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let children: Vec<Scalar> = if norm_t == T::zero() {
            (0..list_size)
                .map(|_| Scalar::zero_value(&element_dtype))
                .collect()
        } else {
            row.iter()
                .map(|&v| Scalar::primitive(v / norm_t, Nullability::NonNullable))
                .collect()
        };

        let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, storage_fsl_nullability);
        let norms_scalar = Scalar::primitive(norm_t, original_nullability);
        (fsl_scalar, norms_scalar)
    });

    let normalized_storage = ConstantArray::new(normalized_fsl_scalar, len).into_array();
    // SAFETY: The single stored row is either `v / ||v||` (unit norm within floating-point
    // tolerance) or all zeros when `||v|| == 0`.
    let normalized = unsafe { NormalizedVector::new_unchecked(normalized_storage) }?;
    let norms_array = ConstantArray::new(norms_scalar, len).into_array();

    // SAFETY: The single stored row is exactly normalized above (or all zeros), and the norm was
    // computed with `sqrt`, so it is non-negative.
    Ok(Some(unsafe {
        L2Denorm::new_array_unchecked(normalized, norms_array, len)?
    }))
}

/// Rebuilds a vector extension array from flat primitive elements.
///
/// # Errors
///
/// Returns an error if the elements are invalid (have incorrect lengths for the
/// `FixedSizeListArray` storage array).
fn build_tensor_array<T: NativePType>(
    dtype: DType,
    tensor_flat_size: usize,
    row_count: usize,
    validity: Validity,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    let storage = build_fsl_storage(tensor_flat_size, row_count, validity, elements)?.into_array();
    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage).into_array())
}

/// Build a non-nullable [`FixedSizeListArray`] suitable for wrapping as a
/// [`NormalizedVector`] storage.
fn build_normalized_storage<T: NativePType>(
    tensor_flat_size: usize,
    row_count: usize,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    Ok(
        build_fsl_storage(tensor_flat_size, row_count, Validity::NonNullable, elements)?
            .into_array(),
    )
}

/// Build a [`FixedSizeListArray`] from a flat element buffer and a per-row validity.
fn build_fsl_storage<T: NativePType>(
    tensor_flat_size: usize,
    row_count: usize,
    validity: Validity,
    elements: Buffer<T>,
) -> VortexResult<FixedSizeListArray> {
    let list_size =
        u32::try_from(tensor_flat_size).vortex_expect("tensor flat size must fit into `u32`");
    // SAFETY: Validity has no length (because vector elements are always non-nullable).
    let elements = unsafe { PrimitiveArray::new_unchecked(elements, Validity::NonNullable) };
    FixedSizeListArray::try_new(elements.into_array(), list_size, validity, row_count)
}

/// Cross-check that `normalized` and `norms` agree on per-row zero-ness, and that stored norms
/// are non-negative. Unit-norm enforcement on the rows lives on the
/// [`NormalizedVector`](crate::normalized_vector::NormalizedVector) dtype itself.
fn validate_norms_against_normalized(
    normalized: &ArrayRef,
    norms: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let vector_metadata = normalized
        .dtype()
        .as_extension_opt()
        .and_then(|ext| ext.metadata_opt::<AnyNormalizedVector>())
        .ok_or_else(|| {
            vortex_err!(
                "L2Denorm normalized child must be a NormalizedVector, got {}",
                normalized.dtype(),
            )
        })?;
    let row_count = normalized.len();
    let element_ptype = vector_metadata.element_ptype();
    let tensor_flat_size = vector_metadata.dimensions() as usize;

    vortex_ensure_eq!(
        norms.len(),
        row_count,
        "L2Denorm normalized and norms children must have the same length"
    );

    let DType::Primitive(norms_ptype, _) = norms.dtype() else {
        vortex_bail!(
            "L2Denorm norms must be a primitive float array, got {}",
            norms.dtype()
        );
    };
    vortex_ensure_eq!(
        *norms_ptype,
        element_ptype,
        "L2Denorm norms ptype must match normalized element ptype"
    );

    if row_count == 0 {
        return Ok(());
    }

    // Drill past the outer `NormalizedVector` wrapper so we always iterate the FSL of the inner
    // plain `Vector`.
    let vector_ref = inner_vector_array(normalized, ctx)?;
    let vector_ext: ExtensionArray = vector_ref.execute(ctx)?;
    let normalized_validity = normalized.validity()?;

    let flat = extract_flat_elements(vector_ext.storage_array(), tensor_flat_size, ctx)?;
    let norms_prim: PrimitiveArray = norms.clone().execute(ctx)?;
    let combined_validity = normalized_validity.and(norms_prim.validity()?)?;

    match_each_float_ptype!(element_ptype, |T| {
        let stored_norms = norms_prim.as_slice::<T>();

        for i in 0..row_count {
            if !combined_validity.is_valid(i)? {
                continue;
            }

            let stored_norm_f64 = ToPrimitive::to_f64(&stored_norms[i]).unwrap_or(f64::NAN);
            vortex_ensure!(
                stored_norm_f64 >= 0.0,
                "L2Denorm norms must be non-negative, but row {i} has {stored_norm_f64:.6}",
            );

            let is_zero_row = flat.row::<T>(i).iter().all(|x| {
                let value = ToPrimitive::to_f64(x).unwrap_or(f64::NAN);
                value == 0.0
            });

            if stored_norm_f64 == 0.0 {
                vortex_ensure!(
                    is_zero_row,
                    "L2Denorm normalized child must be all zeros when norms row {i} is 0.0",
                );
            }
        }
    });

    Ok(())
}

/// Per-operand classification of a tensor argument by whether it carries an authoritative unit-norm
/// representation, and whether stored norms accompany it.
///
/// Symmetric binary tensor operators ([`CosineSimilarity`], [`InnerProduct`]) and unary ones
/// ([`L2Norm`]) take a fast path whenever an operand carries a unit-norm representation. Callers
/// classify each operand individually via [`Self::classify`] and pattern-match on the resulting
/// variant.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
pub(crate) enum NormalForm<'a> {
    /// A plain `Vector`.
    Plain,

    /// An already-normalized `NormalizedVector`, which has implicit norms of `1.0`.
    Normalized { array: &'a ArrayRef },

    /// Decomposed `L2Denorm(normalized, norms)`.
    ///
    /// The normalized child is a `NormalizedVector` by structural contract. It is usually
    /// non-null, with validity stored in `norms`, except when callers use
    /// [`L2Denorm::new_array_unchecked`] directly.
    Denormalized {
        normalized: ArrayRef,
        norms: ArrayRef,
    },
}

impl<'a> NormalForm<'a> {
    /// Classify `array` by its tensor extension type and (if present) any wrapping `L2Denorm`.
    pub(crate) fn classify(array: &'a ArrayRef) -> Self {
        if array.is::<ExactScalarFn<L2Denorm>>() {
            let (normalized, norms) = extract_l2_denorm_children(array);
            return Self::Denormalized { normalized, norms };
        }

        let is_normalized = array
            .dtype()
            .as_extension_opt()
            .is_some_and(|ext| ext.is::<NormalizedVector>());

        if is_normalized {
            Self::Normalized { array }
        } else {
            Self::Plain
        }
    }

    /// Returns the unit-norm "shape" of the operand suitable for inner-product fast paths, if
    /// one exists. For [`Self::Plain`] this returns `None`.
    pub(crate) fn normalized_array(&self) -> Option<&ArrayRef> {
        match self {
            Self::Plain => None,
            Self::Normalized { array } => Some(array),
            Self::Denormalized { normalized, .. } => Some(normalized),
        }
    }
}

#[cfg(test)]
mod tests {

    use rstest::rstest;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::Extension;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::l2_denorm::L2Denorm;
    use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
    use crate::tests::SESSION;
    use crate::types::normalized_vector::NormalizedVector;
    use crate::types::vector::Vector;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::normalized_vector_array;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates L2 denorm on a [`Vector`] (rewrapped as a [`NormalizedVector`]) and the matching
    /// norms, returning the executed array. Convenience wrapper for tests that already hold a
    /// pre-normalized [`Vector`] (possibly wrapped in another encoding such as `MaskedArray`).
    fn eval_l2_denorm(
        vector_input: ArrayRef,
        norms: ArrayRef,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let canonical: ExtensionArray = vector_input.execute(&mut ctx)?;
        let storage = canonical.storage_array().clone();
        let normalized = NormalizedVector::try_new(storage, &mut ctx)?;
        let result = L2Denorm::try_new_array(normalized, norms, len, &mut ctx)?;
        result.into_array().execute(&mut ctx)
    }

    fn non_tensor_extension_array() -> VortexResult<ArrayRef> {
        let storage = PrimitiveArray::from_iter([1i32, 2]).into_array();
        let ext_dtype =
            ExtDType::<Date>::try_new(TimeUnit::Days, storage.dtype().clone())?.erased();
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    fn tensor_snapshot(array: ArrayRef) -> VortexResult<(DType, Vec<bool>, Vec<f64>)> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext: ExtensionArray = array.execute(&mut ctx)?;
        let validity = (0..ext.len())
            .map(|i| ext.is_valid(i, &mut ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
        Ok((
            ext.dtype().clone(),
            validity,
            elements.as_slice::<f64>().to_vec(),
        ))
    }

    fn assert_tensor_arrays_eq(actual: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
        let (actual_dtype, actual_validity, actual_elements) = tensor_snapshot(actual)?;
        let (expected_dtype, expected_validity, expected_elements) = tensor_snapshot(expected)?;

        assert_eq!(actual_dtype, expected_dtype);
        assert_eq!(actual_validity, expected_validity);
        assert_close(&actual_elements, &expected_elements);
        Ok(())
    }

    #[test]
    fn l2_denorm_vectors() -> VortexResult<()> {
        let lhs = vector_array(3, &[0.6, 0.8, 0.0, 0.0, 0.0, 0.0])?;
        let rhs = PrimitiveArray::from_iter([5.0f64, 0.0]).into_array();
        let actual = eval_l2_denorm(lhs, rhs, 2)?;
        let expected = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;

        assert_tensor_arrays_eq(actual, expected)?;
        Ok(())
    }

    #[test]
    fn l2_denorm_null_propagation() -> VortexResult<()> {
        let lhs = vector_array(2, &[0.6, 0.8, 1.0, 0.0, 0.0, 0.0])?;
        let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

        let rhs = PrimitiveArray::from_option_iter([Some(5.0f64), Some(2.0), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let actual: ExtensionArray = eval_l2_denorm(lhs, rhs, 3)?.execute(&mut ctx)?;
        let storage: FixedSizeListArray = actual.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

        assert!(actual.is_valid(0, &mut ctx)?);
        assert!(!actual.is_valid(1, &mut ctx)?);
        assert!(!actual.is_valid(2, &mut ctx)?);
        assert_close(&elements.as_slice::<f64>()[..2], &[3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_non_extension_lhs() {
        let lhs = PrimitiveArray::from_iter([1.0f64, 2.0]).into_array();
        let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
    }

    #[test]
    fn l2_denorm_rejects_non_tensor_extension_lhs() -> VortexResult<()> {
        let lhs = non_tensor_extension_array()?;
        let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_plain_unit_vector_lhs() -> VortexResult<()> {
        let lhs = vector_array(2, &[1.0, 0.0, 0.0, 1.0])?;
        let rhs = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_unnormalized_plain_vector_lhs() -> VortexResult<()> {
        let lhs = vector_array(2, &[3.0, 4.0, 0.0, 1.0])?;
        let rhs = PrimitiveArray::from_iter([5.0f64, 1.0]).into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_mismatched_rhs_ptype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_vector_array(2, &[1.0, 0.0, 0.0, 1.0], &mut ctx)?;
        let rhs = PrimitiveArray::from_iter([1.0f32, 1.0]).into_array();

        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_non_primitive_rhs_without_panic() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_vector_array(2, &[1.0, 0.0, 0.0, 1.0], &mut ctx)?;
        let rhs = vector_array(2, &[1.0f64, 0.0, 0.0, 1.0])?;

        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_rejects_length_mismatch_without_panic() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = normalized_vector_array(2, &[1.0, 0.0, 0.0, 1.0], &mut ctx)?;
        let rhs = PrimitiveArray::from_iter([1.0f64]).into_array();

        let result = L2Denorm::try_new_array(lhs, rhs, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn normalized_vector_try_new_accepts_normalized_f16_input() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = [3.0f32, 4.0, 0.0, 0.0].map(half::f16::from_f32);
        let roundtrip = normalize_as_l2_denorm(vector_array(2, &elements)?, &mut ctx)?;
        // The first child is already a `NormalizedVector` by construction.
        let normalized = roundtrip.child_at(0).clone();
        assert!(normalized.dtype().as_extension().is::<NormalizedVector>(),);
        Ok(())
    }

    #[test]
    fn normalized_vector_try_new_rejects_unnormalized_input() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = normalized_vector_array(2, &[3.0, 4.0, 1.0, 0.0], &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_try_new_array_rejects_nonzero_row_with_zero_norm() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let normalized = normalized_vector_array(2, &[1.0, 0.0, 0.0, 0.0], &mut ctx)?;
        let norms = PrimitiveArray::from_iter([0.0f64, 0.0]).into_array();

        let result = L2Denorm::try_new_array(normalized, norms, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_try_new_array_rejects_negative_norms() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let normalized = normalized_vector_array(2, &[1.0, 0.0, 0.0, 1.0], &mut ctx)?;
        let norms = PrimitiveArray::from_iter([1.0f64, -1.0]).into_array();

        let result = L2Denorm::try_new_array(normalized, norms, 2, &mut ctx);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn l2_denorm_new_array_unchecked_skips_zero_row_cross_check() -> VortexResult<()> {
        // `L2Denorm::new_array_unchecked` accepts a NormalizedVector + norms without the zero-norm
        // cross-check; useful for lossy encodings (e.g. TurboQuant).
        let mut ctx = SESSION.create_execution_ctx();
        let normalized = normalized_vector_array(2, &[1.0, 0.0, 0.0, 1.0], &mut ctx)?;
        let norms = PrimitiveArray::from_iter([0.0f64, 1.0]).into_array();

        // SAFETY: This test intentionally exercises the lossy escape hatch.
        let result = unsafe { L2Denorm::new_array_unchecked(normalized, norms, 2) };
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn l2_denorm_new_array_unchecked_rejects_plain_vector_lhs() -> VortexResult<()> {
        let vector = vector_array(2, &[1.0, 0.0, 0.0, 1.0])?;
        let norms = PrimitiveArray::from_iter([1.0f64, 1.0]).into_array();

        // SAFETY: This deliberately checks that structural validation still rejects a plain
        // `Vector` child.
        let result = unsafe { L2Denorm::new_array_unchecked(vector, norms, 2) };
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_roundtrips_vectors() -> VortexResult<()> {
        let input = vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0])?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_rejects_fixed_shape_tensor() -> VortexResult<()> {
        let input = tensor_array(&[2, 2], &[3.0, 4.0, 0.0, 0.0])?;
        let mut ctx = SESSION.create_execution_ctx();

        assert!(normalize_as_l2_denorm(input, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_supports_constant_vectors() -> VortexResult<()> {
        let input = Vector::constant_array(&[3.0, 4.0], 2)?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_constant_input_has_constant_children() -> VortexResult<()> {
        // The constant fast path in `normalize_as_l2_denorm` must produce an `L2Denorm` whose
        // normalized storage and norms child are both still `ConstantArray`s. This is what
        // allows downstream ops (cosine similarity, inner product) to short-circuit.
        let input = Vector::constant_array(&[3.0, 4.0], 16)?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(input, &mut ctx)?;

        // The normalized child is a `NormalizedVector(Vector(Constant<FSL>))`. Drill past both
        // extension layers and confirm the innermost FSL storage is still constant-backed.
        let normalized = roundtrip.child_at(0).clone();
        let normalized_ext = normalized
            .as_opt::<Extension>()
            .expect("normalized child should be an Extension array");
        let inner_vector = normalized_ext
            .storage_array()
            .as_opt::<Extension>()
            .expect("NormalizedVector storage should be a Vector extension array");
        assert!(
            inner_vector.storage_array().as_opt::<Constant>().is_some(),
            "normalized storage should stay constant after the fast path"
        );

        // The norms child must itself be a ConstantArray with the exact precomputed norm.
        let norms = roundtrip.child_at(1).clone();
        let norms_const = norms
            .as_opt::<Constant>()
            .expect("norms child should be a ConstantArray");
        assert_close(
            &[norms_const
                .scalar()
                .as_primitive()
                .typed_value::<f64>()
                .expect("norms scalar")],
            &[5.0],
        );
        Ok(())
    }

    #[test]
    fn normalize_as_l2_denorm_uses_zero_rows_for_zero_norms() -> VortexResult<()> {
        let input = vector_array(2, &[0.0, 0.0, 3.0, 4.0])?;
        let mut ctx = SESSION.create_execution_ctx();
        let roundtrip = normalize_as_l2_denorm(input.clone(), &mut ctx)?;
        // Normalized child is a `NormalizedVector` wrapping a `Vector` wrapping the FSL; drill
        // past the outer `NormalizedVector` to reach the underlying `Vector`.
        let normalized: ExtensionArray = roundtrip.child_at(0).clone().execute(&mut ctx)?;
        let vector: ExtensionArray = normalized.storage_array().clone().execute(&mut ctx)?;
        let storage: FixedSizeListArray = vector.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
        let actual = roundtrip.into_array().execute(&mut ctx)?;

        assert_close(&elements.as_slice::<f64>()[..2], &[0.0, 0.0]);
        assert_tensor_arrays_eq(actual, input)?;
        Ok(())
    }

    /// `NormalizedVector` input takes the fast path: re-applying norms must reconstruct the
    /// original element values bit-for-bit (since the divisor in the slow path would be 1.0
    /// for unit rows and 0.0 for zero rows).
    #[test]
    fn normalize_as_l2_denorm_normalized_vector_round_trips_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = [0.6, 0.8, 0.0, 0.0, 1.0, 0.0];
        let input = normalized_vector_array(2, &elements, &mut ctx)?;

        let roundtrip = normalize_as_l2_denorm(input, &mut ctx)?;
        let executed: ExtensionArray = roundtrip.into_array().execute(&mut ctx)?;
        let storage: FixedSizeListArray = executed.storage_array().clone().execute(&mut ctx)?;
        let executed_elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;

        assert_close(executed_elements.as_slice::<f64>(), &elements);
        Ok(())
    }

    /// The `NormalizedVector` fast path borrows `L2Norm`'s short-circuit, which emits `1.0` for
    /// unit rows and `0.0` for zero rows. Tag the zero row with norm `0.0` here (not `1.0`) so a
    /// downstream `L2Norm` over the resulting `L2Denorm` continues to read the right value.
    #[test]
    fn normalize_as_l2_denorm_normalized_vector_emits_unit_or_zero_norms() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = normalized_vector_array(2, &[0.6, 0.8, 0.0, 0.0, 1.0, 0.0], &mut ctx)?;

        let l2_denorm = normalize_as_l2_denorm(input, &mut ctx)?;
        let norms: PrimitiveArray = l2_denorm.child_at(1).clone().execute(&mut ctx)?;

        assert_close(norms.as_slice::<f64>(), &[1.0, 0.0, 1.0]);
        Ok(())
    }

    /// The `NormalizedVector` fast path returns the input unchanged as the `normalized` child
    /// rather than re-allocating storage to satisfy the slow path's "always non-nullable"
    /// invariant. Verify that the child dtype is still a `NormalizedVector` extension after the
    /// fast path.
    #[test]
    fn normalize_as_l2_denorm_normalized_vector_preserves_normalized_child_dtype()
    -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = normalized_vector_array(2, &[0.6, 0.8, 1.0, 0.0], &mut ctx)?;

        let l2_denorm = normalize_as_l2_denorm(input, &mut ctx)?;
        let normalized = l2_denorm.child_at(0).clone();

        assert!(normalized.dtype().as_extension().is::<NormalizedVector>());
        Ok(())
    }

    /// Builds a non-nullable constant f64 norms array of length `len`.
    fn constant_f64_norms(value: f64, len: usize) -> ArrayRef {
        ConstantArray::new(Scalar::primitive(value, Nullability::NonNullable), len).into_array()
    }

    #[test]
    fn l2_denorm_constant_unit_norms_is_noop() -> VortexResult<()> {
        // Every stored norm is exactly 1.0, so the constant fast path must short-circuit and
        // return the normalized child unchanged.
        let normalized = vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
        let norms = constant_f64_norms(1.0, 2);

        let actual = eval_l2_denorm(normalized.clone(), norms, 2)?;
        assert_tensor_arrays_eq(actual, normalized)?;
        Ok(())
    }

    #[test]
    fn l2_denorm_constant_near_unit_norms_is_noop() -> VortexResult<()> {
        // A norm that differs from 1.0 by less than the f64 unit-norm tolerance must still
        // hit the fast path and return the normalized child unchanged.
        let normalized = vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
        let norms = constant_f64_norms(1.0 + 1e-12, 2);

        let actual = eval_l2_denorm(normalized.clone(), norms, 2)?;
        assert_tensor_arrays_eq(actual, normalized)?;
        Ok(())
    }

    #[test]
    fn l2_denorm_constant_nonunit_norms_scales_vectors() -> VortexResult<()> {
        // A constant norm that is not 1.0 must scale every element of every row by the same
        // factor via the backing elements multiplication path.
        let normalized = vector_array(3, &[0.6, 0.8, 0.0, 1.0, 0.0, 0.0])?;
        let norms = constant_f64_norms(5.0, 2);

        let actual = eval_l2_denorm(normalized, norms, 2)?;
        let expected = vector_array(3, &[3.0, 4.0, 0.0, 5.0, 0.0, 0.0])?;
        assert_tensor_arrays_eq(actual, expected)?;
        Ok(())
    }

    /// Regression: a non-nullable [`NormalizedVector`] child paired with a nullable-dtype
    /// constant norms array (whose value happens to be non-null `1.0`) used to fail in the
    /// constant-unit fast path because the extension's declared storage nullability no longer
    /// matched the storage array's own nullability.
    #[test]
    fn l2_denorm_constant_unit_norms_nullable_scalar_nonnullable_normalized() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let normalized = normalized_vector_array(3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &mut ctx)?;
        let nullable_unit_norms =
            ConstantArray::new(Scalar::primitive(1.0f64, Nullability::Nullable), 2).into_array();

        let result = L2Denorm::try_new_array(normalized, nullable_unit_norms, 2, &mut ctx)?;
        let actual: ArrayRef = result.into_array().execute(&mut ctx)?;

        // The output surfaces as a plain `Vector` whose outer nullability is the union of the
        // two children (nullable here, since the norms child was nullable).
        assert!(actual.dtype().as_extension().is::<Vector>());
        assert!(actual.dtype().is_nullable());

        // The element values round-trip: scaling unit vectors by `1.0` is a no-op.
        let ext: ExtensionArray = actual.execute(&mut ctx)?;
        let storage: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;
        let elements: PrimitiveArray = storage.elements().clone().execute(&mut ctx)?;
        assert_close(elements.as_slice::<f64>(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        Ok(())
    }

    /// Build an `L2Denorm` array from a raw input (which may have nullable storage) by running
    /// `normalize_as_l2_denorm`. The normalized child ends up non-nullable, and the norms child
    /// inherits the input's nullability, giving us two different per-child nullabilities to
    /// round-trip.
    #[rstest]
    #[case::vector(vector_array(3, &[3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).unwrap())]
    fn serde_round_trip(#[case] input: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let original = normalize_as_l2_denorm(input, &mut ctx)?.into_array();

        let scalar_fn_array = original.as_::<vortex_array::arrays::ScalarFn>();
        let children = scalar_fn_array.children();

        let plugin = ScalarFnArrayPlugin::new(L2Denorm);
        let metadata = plugin
            .serialize(&original, &SESSION)?
            .expect("L2Denorm serialize must produce metadata");

        let recovered = plugin.deserialize(
            original.dtype(),
            original.len(),
            &metadata,
            &[],
            &children,
            &SESSION,
        )?;

        assert_eq!(recovered.dtype(), original.dtype());
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered.encoding_id(), original.encoding_id());
        Ok(())
    }
}
