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
use vortex_array::builtins::ArrayBuiltins;
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

use crate::encodings::normalized::Normalized;
use crate::encodings::normalized::NormalizedArray;
use crate::encodings::normalized::NormalizedArraySlotsExt;
use crate::encodings::normalized::NormalizedSlots;
use crate::encodings::normalized::array::DATA_CHILDREN;
use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::utils::extract_constant_flat_row;
use crate::utils::extract_flat_elements;
use crate::utils::validate_tensor_float_input;

/// The compression scheme that rewrites a tensor-like column into the [`Normalized`] encoding.
#[derive(Debug)]
pub struct NormalizedScheme;

impl Scheme for NormalizedScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.tensor.normalized"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext) = canonical else {
            return false;
        };

        // `AlwaysUse` prevents later schemes from seeing a claimed array, so match only the float
        // tensor dtypes accepted by `compress`.
        ext.ext_dtype()
            .metadata_opt::<AnyTensor>()
            .is_some_and(|tensor| tensor.element_ptype().is_float())
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Normalized.id()]
    }

    fn num_children(&self) -> usize {
        // Only the two data children are required; validity is optional.
        DATA_CHILDREN
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
        let normalized_array = normalize(data.array().clone(), exec_ctx)?;

        // Splitting magnitude out is only worth anything if the children then compress: the
        // unit-norm coordinates have a bounded range and the norms are an ordinary float column.
        let normalized = compressor.compress_child(
            normalized_array.normalized(),
            &compress_ctx,
            self.id(),
            NormalizedSlots::NORMALIZED,
            exec_ctx,
        )?;
        let norms = compressor.compress_child(
            normalized_array.norms(),
            &compress_ctx,
            self.id(),
            NormalizedSlots::NORMS,
            exec_ctx,
        )?;

        let validity = normalized_array.validity()?;

        // SAFETY: Cascading preserves the split's child lengths and dtypes, and the validity is
        // carried over from the split unchanged.
        Ok(unsafe { Normalized::new_unchecked(normalized, norms, validity) }.into_array())
    }
}

/// Splits a tensor-like column into its exact [`Normalized`] representation.
///
/// The children are non-nullable, and the input validity moves to the parent. Both children are
/// zero at null rows so masked physical values cannot reach downstream encodings.
///
/// # Errors
///
/// Returns an error if `input` is not a float tensor column or if execution fails.
pub fn normalize(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<NormalizedArray> {
    let row_count = input.len();
    let tensor_match = validate_tensor_float_input(input.dtype())?;
    let tensor_flat_size = tensor_match.list_size() as usize;

    // Normalize a constant's stored row without expanding it to the column length.
    if let Some(wrapped) = try_build_constant_normalized(&input, row_count, ctx)? {
        return Ok(wrapped);
    }

    let norms_array: ArrayRef = L2Norm
        .try_new_array(row_count, EmptyOptions, [input.clone()])?
        .execute(ctx)?;

    // Execute before reading validity. Reading validity from the lazy array would run `L2Norm`
    // again when the values are requested.
    let primitive_norms: PrimitiveArray = norms_array.execute(ctx)?;

    let validity = primitive_norms.validity()?;

    // Skip `fill_null` for non-nullable input because it still executes a cast.
    let norms: PrimitiveArray = if validity.nullability().is_nullable() {
        let element_dtype =
            DType::Primitive(tensor_match.element_ptype(), Nullability::NonNullable);

        primitive_norms
            .into_array()
            .fill_null(Scalar::zero_value(&element_dtype))?
            .execute(ctx)?
    } else {
        primitive_norms
    };

    let input: ExtensionArray = input.execute(ctx)?;
    let normalized_dtype = input.dtype().as_nonnullable();
    let flat = extract_flat_elements(input.storage_array(), tensor_flat_size, ctx)?;

    let normalized = match_each_float_ptype!(flat.ptype(), |T| {
        let norm_values = norms.as_slice::<T>();

        let total_elements = row_count * tensor_flat_size;
        let mut elements = BufferMut::<T>::with_capacity(total_elements);
        for i in 0..row_count {
            let norm = norm_values[i];

            // SAFETY: We allocated `row_count * tensor_flat_size` capacity and push exactly
            // `tensor_flat_size` elements per row.
            if norm.is_zero() {
                unsafe { elements.push_n_unchecked(T::zero(), tensor_flat_size) };
            } else {
                for &x in flat.row::<T>(i) {
                    unsafe { elements.push_unchecked(x / norm) };
                }
            }
        }

        build_normalized(
            normalized_dtype,
            tensor_flat_size,
            row_count,
            elements.freeze(),
        )
    })?;

    // SAFETY: This split creates non-nullable children with matching lengths and element ptypes.
    // The captured validity describes the same input rows.
    Ok(unsafe { Normalized::new_unchecked(normalized, norms.into_array(), validity) })
}

/// Normalizes a single constant row without expanding it to the column length.
///
/// Returns `Ok(None)` unless `input` has a non-null constant fixed-size-list storage scalar. A
/// matching input produces constant normalized and norms children.
pub(crate) fn try_build_constant_normalized(
    input: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<NormalizedArray>> {
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

    let tensor_match = input
        .dtype()
        .as_extension()
        .metadata_opt::<AnyTensor>()
        .vortex_expect("caller validated input has AnyTensor metadata");
    let list_size = tensor_match.list_size() as usize;

    // A non-null constant with a nullable dtype is all-valid.
    let validity = Validity::from(input.dtype().nullability());
    let normalized_ext_dtype = input.dtype().as_nonnullable().as_extension().clone();

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
        // This mirrors the per-row logic in `normalize`.
        let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let children: Vec<Scalar> = if norm_t.is_zero() {
            (0..list_size)
                .map(|_| Scalar::zero_value(&element_dtype))
                .collect()
        } else {
            row.iter()
                .map(|&v| Scalar::primitive(v / norm_t, Nullability::NonNullable))
                .collect()
        };

        let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
        let norms_scalar = Scalar::primitive(norm_t, Nullability::NonNullable);
        (fsl_scalar, norms_scalar)
    });

    let normalized_storage = ConstantArray::new(normalized_fsl_scalar, len).into_array();
    let normalized = ExtensionArray::new(normalized_ext_dtype, normalized_storage).into_array();
    let norms = ConstantArray::new(norms_scalar, len).into_array();

    // SAFETY: Both constants use `len`, are non-nullable, and have the input element ptype. The
    // validity comes from the same input column.
    Ok(Some(unsafe {
        Normalized::new_unchecked(normalized, norms, validity)
    }))
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
