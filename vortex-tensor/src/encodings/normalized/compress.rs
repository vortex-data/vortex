// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::CompressorContext;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::encodings::normalized::Normalized;
use crate::encodings::normalized::NormalizedArray;
use crate::encodings::normalized::NormalizedArraySlotsExt;
use crate::encodings::normalized::NormalizedSlots;
use crate::encodings::normalized::array::DATA_CHILDREN;
use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_normalize::normalize_children;
use crate::scalar_fns::l2_normalize::normalize_row_into;
use crate::scalar_fns::l2_normalize::normalized_output_dtype;
use crate::types::unit_vector::AnyUnitVector;
use crate::types::unit_vector::UnitVector;
use crate::utils::extract_constant_flat_row;

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
        !ext.ext_dtype().is::<UnitVector>()
            && ext
                .ext_dtype()
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
        let dtype = data.array().dtype().clone();
        // A tensor can be valid even when its norm cannot be represented in its element ptype.
        // Compression must preserve the original array for these value-dependent failures.
        let normalized_array = match normalize(data.array().clone(), exec_ctx) {
            Ok(normalized) => normalized,
            Err(VortexError::InvalidArgument(..)) => return Ok(data.array().clone()),
            Err(error) => return Err(error),
        };

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
        Ok(
            unsafe { Normalized::new_unchecked_with_dtype(dtype, normalized, norms, validity) }
                .into_array(),
        )
    }
}

/// Splits a tensor-like column into its exact [`Normalized`] representation.
///
/// The children are non-nullable, and the input validity moves to the parent. Both children are
/// zero at null rows so masked physical values cannot reach downstream encodings.
///
/// # Errors
///
/// Returns an error if `input` is not an ordinary float Vector or float FixedShapeTensor, if a row
/// cannot be normalized in its element ptype, or if execution fails.
pub fn normalize(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<NormalizedArray> {
    vortex_ensure!(
        !input
            .dtype()
            .as_extension_opt()
            .is_some_and(|dtype| dtype.is::<AnyUnitVector>()),
        InvalidArgument: "Normalized input must be an ordinary Vector or FixedShapeTensor, got {}",
        input.dtype(),
    );

    if let Some(normalized) = try_build_constant_normalized(&input, ctx)? {
        return Ok(normalized);
    }

    let dtype = input.dtype().clone();
    let (normalized, norms, validity) = normalize_children(input, ctx)?;

    // SAFETY: `normalize_children` constructs compatible non-nullable children and carries the
    // input validity.
    Ok(unsafe { Normalized::new_unchecked_with_dtype(dtype, normalized, norms, validity) })
}

/// Normalizes a single constant row without expanding it to the column length.
///
/// Returns `Ok(None)` unless `input` has a non-null constant tensor row.
pub(crate) fn try_build_constant_normalized(
    input: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<NormalizedArray>> {
    if input
        .dtype()
        .as_extension_opt()
        .is_some_and(|dtype| dtype.is::<AnyUnitVector>())
    {
        return Ok(None);
    }

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

    if input
        .dtype()
        .as_extension()
        .metadata_opt::<AnyTensor>()
        .is_none()
    {
        return Ok(None);
    }

    // A non-null constant with a nullable dtype is all-valid.
    let validity = Validity::from(input.dtype().nullability());
    let normalized_ext_dtype = normalized_output_dtype(input.dtype())?
        .as_extension()
        .clone();

    let flat = extract_constant_flat_row(storage, ctx)?;

    let scalars = match_each_float_ptype!(flat.ptype(), |T| {
        let row = flat.as_slice::<T>();
        let mut normalized = BufferMut::<T>::with_capacity(row.len());
        // SAFETY: `normalized` reserves space for the entire row.
        let norm = unsafe { normalize_row_into(row, &mut normalized)? };
        let element_dtype = DType::Primitive(T::PTYPE, Nullability::NonNullable);
        let children = normalized
            .freeze()
            .iter()
            .copied()
            .map(|value| Scalar::primitive(value, Nullability::NonNullable))
            .collect();

        let fsl_scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
        let norms_scalar = Scalar::primitive(norm, Nullability::NonNullable);
        Ok::<_, VortexError>((fsl_scalar, norms_scalar))
    });
    // This is also an optimization for infallible scalar functions. Value-dependent failure must
    // fall back to their ordinary execution path.
    let Ok((normalized_fsl_scalar, norms_scalar)) = scalars else {
        return Ok(None);
    };

    let len = input.len();
    let normalized_storage = ConstantArray::new(normalized_fsl_scalar, len).into_array();
    let normalized = if normalized_ext_dtype.is::<UnitVector>() {
        // SAFETY: The stored row was produced by `normalize_row_into` and the constant repeats it.
        unsafe { UnitVector::new_unchecked(normalized_storage)? }
    } else {
        ExtensionArray::new(normalized_ext_dtype, normalized_storage).into_array()
    };
    let norms = ConstantArray::new(norms_scalar, len).into_array();

    // SAFETY: Both constants use `len`, are non-nullable, and have the input element ptype. The
    // validity comes from the same input column.
    Ok(Some(unsafe {
        Normalized::new_unchecked_with_dtype(input.dtype().clone(), normalized, norms, validity)
    }))
}
