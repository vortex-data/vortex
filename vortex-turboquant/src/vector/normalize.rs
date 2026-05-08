// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant-local vector normalization.

// TODO(connor): Remove this comment once we delete the other version in `vortex-tensor`.
// The tensor crate also has a `normalize_as_l2_denorm` helper, but TurboQuant needs different
// validity semantics: a null vector is not a zero vector, so invalid rows keep their row validity
// on both `L2Denorm` children and downstream quantization skips them.

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_tensor::scalar_fns::l2_denorm::L2Denorm;
use vortex_tensor::scalar_fns::l2_norm::L2Norm;
use vortex_tensor::vector::AnyVector;
use vortex_tensor::vector::Vector;

/// Normalize a `Vector` array and wrap it with its original row norms with [`L2Denorm`].
///
/// This preserves input row validity on both [`L2Denorm`] children. Or in other words, validity is
/// propagated down to the children so that TurboQuant can skip quantizing those vectors (as it does
/// not have a good way to represent 0 vectors in its quantized domain).
pub(crate) fn tq_normalize_as_l2_denorm(
    input: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnArray> {
    let row_count = input.len();
    let vector_metadata = input
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
        .ok_or_else(|| vortex_err!("TurboQuant normalization expects a Vector extension array"))?;
    let dimensions = vector_metadata.dimensions() as usize;
    let vector_validity = input.validity()?;

    // Use `L2Norm` to calculate the normals for each vector.
    let norms: ArrayRef = L2Norm::try_new_array(input.clone(), row_count)?
        .into_array()
        .execute(ctx)?;
    let primitive_norms: PrimitiveArray = norms.clone().execute(ctx)?;

    let input: ExtensionArray = input.execute(ctx)?;
    let storage: FixedSizeListArray = input.storage_array().clone().execute(ctx)?;
    vortex_ensure_eq!(
        storage.list_size() as usize,
        dimensions,
        "Vector storage dimension must be {dimensions}, got {}",
        storage.list_size()
    );
    let elements: PrimitiveArray = storage.elements().clone().execute(ctx)?;

    let mask = vector_validity.execute_mask(row_count, ctx)?;

    let normalized = match_each_float_ptype!(elements.ptype(), |T| {
        normalize_vectors::<T>(
            &elements,
            &primitive_norms,
            &mask,
            dimensions,
            vector_validity.clone(),
        )
    })?;

    // SAFETY: matches the lossy-encoding relaxation documented on
    // [`L2Denorm::new_array_unchecked`]. Norms come from `L2Norm` over the same input, so they
    // match the vector element type and row count. Valid nonzero rows are divided by their stored
    // norm and are unit-norm. Valid zero-norm rows and invalid rows use physical zero placeholders;
    // invalid rows remain guarded by row-level invalid validity.
    unsafe { L2Denorm::new_array_unchecked(normalized, norms, row_count) }
}

fn normalize_vectors<T>(
    elements: &PrimitiveArray,
    norms: &PrimitiveArray,
    mask: &Mask,
    dimensions: usize,
    vector_validity: Validity,
) -> VortexResult<ArrayRef>
where
    T: Float + NativePType,
{
    let num_vectors = norms.len();

    let values = elements.as_slice::<T>();
    let norm_values = norms.as_slice::<T>();

    let output_len = num_vectors
        .checked_mul(dimensions)
        .ok_or_else(|| vortex_err!("TurboQuant normalized vector length overflow"))?;
    let mut output = BufferMut::<T>::with_capacity(output_len);

    for i in 0..num_vectors {
        let row_values = &values[i * dimensions..][..dimensions];
        let norm = norm_values[i];

        if !mask.value(i) || norm == T::zero() {
            // Invalid rows are placeholders guarded by validity. A valid vector with L2 norm zero
            // is all zeros. In both cases, append zero placeholders without per-element work.

            // SAFETY: `output` was allocated with capacity `output_len`, and the loop appends
            // exactly `dimensions` values for each of `num_vectors` iterations.
            unsafe { output.push_n_unchecked(T::zero(), dimensions) };
        } else {
            for &value in row_values {
                // SAFETY: same capacity invariant as above.
                unsafe { output.push_unchecked(value / norm) };
            }
        }
    }

    // Vector elements are always non-nullable.
    let elements = PrimitiveArray::new::<T>(output.freeze(), Validity::NonNullable);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "this initially came from a u32"
    )]
    let storage = FixedSizeListArray::try_new(
        elements.into_array(),
        dimensions as u32,
        vector_validity,
        num_vectors,
    )?;

    Ok(
        ExtensionArray::try_new_from_vtable(Vector, EmptyMetadata, storage.into_array())?
            .into_array(),
    )
}
