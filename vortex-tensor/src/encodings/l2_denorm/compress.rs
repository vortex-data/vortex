// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Float;
use num_traits::Zero;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::CompressorContext;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::encodings::l2_denorm::L2Denorm;
use crate::encodings::l2_denorm::L2DenormArray;
use crate::encodings::l2_denorm::L2DenormArraySlotsExt;
use crate::encodings::l2_denorm::L2DenormSlots;
use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::utils::extract_constant_flat_row;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// The compression scheme that rewrites a tensor-like column into the [`L2Denorm`] encoding.
#[derive(Debug)]
pub struct L2DenormScheme;

impl Scheme for L2DenormScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.tensor.l2_denorm"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(
            canonical,
            Canonical::Extension(ext) if ext.ext_dtype().is::<AnyTensor>()
        )
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![L2Denorm.id()]
    }

    /// Children: normalized=0, norms=1.
    fn num_children(&self) -> usize {
        L2DenormSlots::COUNT
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let denorm = normalize_as_l2_denorm(data.array().clone(), exec_ctx)?;

        // Splitting magnitude out is only worth anything if the children then compress: the
        // unit-norm coordinates have a bounded range and the norms are an ordinary float column.
        let normalized = compressor.compress_child(
            denorm.normalized(),
            &compress_ctx,
            self.id(),
            L2DenormSlots::NORMALIZED,
            exec_ctx,
        )?;
        let norms = compressor.compress_child(
            denorm.norms(),
            &compress_ctx,
            self.id(),
            L2DenormSlots::NORMS,
            exec_ctx,
        )?;

        // SAFETY: Cascading preserves the split's child lengths and dtypes.
        Ok(unsafe { L2Denorm::new_unchecked(normalized, norms) }.into_array())
    }
}

/// Splits a tensor-like column into its exact [`L2Denorm`] representation.
///
/// # Normalized child
///
/// The normalized child is always **non-nullable**. Every non-null row with a positive L2 norm is
/// divided by its norm to produce a unit-norm row.
///
/// Rows that are null in the original input are **zeroed out** in the normalized output. Null rows
/// may carry undefined physical storage values, and we do not want that garbage propagating into
/// downstream lossy encodings of the normalized child.
///
/// # Nullability
///
/// Nullability is tracked entirely by the norms child, which inherits the input's nulls through
/// [`L2Norm`]'s validity propagation. The [`L2Denorm`] array's validity is the `and` of both
/// children, so an all-valid normalized child plus a nullable norms child reproduces the input's
/// validity exactly.
///
/// Because this computes exact norms first and then divides by them, the returned `normalized`
/// child satisfies the strict unit-norm invariant.
pub fn normalize_as_l2_denorm(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<L2DenormArray> {
    let row_count = input.len();
    let tensor_match = validate_tensor_float_input(input.dtype())?;
    let tensor_flat_size = tensor_match.list_size() as usize;

    // Constant fast path: if the input is a constant-backed extension, normalize the single stored
    // row once and return an `L2Denorm` whose children are both `ConstantArray`s.
    if let Some(wrapped) = try_build_constant_l2_denorm(&input, row_count, ctx)? {
        return Ok(wrapped);
    }

    let norms_array: ArrayRef = L2Norm
        .try_new_array(row_count, EmptyOptions, [input.clone()])?
        .execute(ctx)?;
    let primitive_norms: PrimitiveArray = norms_array.clone().execute(ctx)?;
    let norms_validity = primitive_norms.validity()?;

    let input: ExtensionArray = input.execute(ctx)?;
    let normalized_dtype = input.dtype().as_nonnullable();
    let flat = extract_flat_elements(input.storage_array(), tensor_flat_size, ctx)?;

    // Resolve validity to a mask once rather than probing it per row (each `Validity::is_valid`
    // executes a scalar for array-backed validity).
    let norms_valid = norms_validity.execute_mask(row_count, ctx)?;

    let normalized = match_each_float_ptype!(flat.ptype(), |T| {
        let norm_values = primitive_norms.as_slice::<T>();

        let total_elements = row_count * tensor_flat_size;
        let mut elements = BufferMut::<T>::with_capacity(total_elements);
        for i in 0..row_count {
            let is_valid = norms_valid.value(i);
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

        // Since L2Denorm's validity is the `and` of its child validities, the normalized child can
        // be non-nullable.
        build_normalized(
            normalized_dtype,
            tensor_flat_size,
            row_count,
            elements.freeze(),
        )
    })?;

    // SAFETY: The normalized rows, norms ptype, and child lengths come directly from this split.
    Ok(unsafe { L2Denorm::new_unchecked(normalized, norms_array) })
}

/// Attempts to build an [`L2DenormArray`] whose two children are both [`ConstantArray`]s by
/// eagerly normalizing `input`'s single stored row.
///
/// Returns `Ok(None)` when `input` is not a tensor-like extension array whose storage is a
/// [`ConstantArray`] with a non-null fixed-size-list scalar.
///
/// When `input` matches, the result is equivalent to [`normalize_as_l2_denorm`] but runs in
/// `O(list_size)` instead of `O(row_count * list_size)`. Keeping both children constant is what
/// lets cosine similarity and inner product short-circuit against a literal query vector.
pub(crate) fn try_build_constant_l2_denorm(
    input: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<L2DenormArray>> {
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

    // The caller is expected to have already validated that `input` is an `AnyTensor` extension
    // dtype.
    let tensor_match = input
        .dtype()
        .as_extension()
        .metadata_opt::<AnyTensor>()
        .vortex_expect("caller validated input has AnyTensor metadata");
    let list_size = tensor_match.list_size() as usize;
    let original_nullability = input.dtype().nullability();
    let ext_dtype = input.dtype().as_extension().clone();
    let storage_fsl_nullability = storage.dtype().nullability();

    // Materialize just the single stored row; this does not expand the constant to the full column
    // length.
    let flat = extract_constant_flat_row(storage, ctx)?;

    let (normalized_fsl_scalar, norms_scalar) = match_each_float_ptype!(flat.ptype(), |T| {
        let row = flat.as_slice::<T>();

        let mut sum_sq = T::zero();
        for &x in row {
            sum_sq += x * x;
        }
        let norm_t: T = sum_sq.sqrt();

        // Zero-norm rows must be stored as all-zeros so the unit-norm-or-zero invariant holds.
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

        // The rebuilt FSL scalar preserves the original storage FSL's nullability so the resulting
        // `ExtensionArray::new` call accepts the same extension dtype.
        let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, storage_fsl_nullability);
        let norms_scalar = Scalar::primitive(norm_t, original_nullability);
        (fsl_scalar, norms_scalar)
    });

    let normalized_storage = ConstantArray::new(normalized_fsl_scalar, len).into_array();
    let normalized = ExtensionArray::new(ext_dtype, normalized_storage).into_array();
    let norms = ConstantArray::new(norms_scalar, len).into_array();

    // SAFETY: The constant children have matching lengths and element ptypes.
    Ok(Some(unsafe { L2Denorm::new_unchecked(normalized, norms) }))
}

/// Builds the non-nullable tensor-like extension array that becomes the `normalized` child.
fn build_normalized<T: NativePType>(
    dtype: DType,
    tensor_flat_size: usize,
    row_count: usize,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    let list_size =
        u32::try_from(tensor_flat_size).vortex_expect("tensor flat size must fit into `u32`");

    // SAFETY: Tensor elements are always non-nullable, so the validity carries no length.
    let elements = unsafe { PrimitiveArray::new_unchecked(elements, Validity::NonNullable) };

    let storage = FixedSizeListArray::try_new(
        elements.into_array(),
        list_size,
        Validity::NonNullable,
        row_count,
    )?;

    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage.into_array()).into_array())
}
