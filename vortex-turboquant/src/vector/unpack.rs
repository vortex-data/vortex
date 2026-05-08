// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant unpacking (dequantization) logic.
//!
//! Note that because TurboQuant is a lossy compression scheme, unpacking does not roundtrip with
//! the initial packing.

use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_float_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_tensor::vector::Vector;

use super::storage::parse_storage;
use super::tq_padded_dim;
use crate::centroids::compute_or_get_centroids;
use crate::sorf::SorfMatrix;
use crate::vtable::TurboQuantMetadata;

/// Decode a `TurboQuant` extension array back into a `Vector` extension array.
///
/// The decoded directions are inverse-transformed, truncated to the original dimension, and
/// multiplied by the stored row norms. The conversion is lossy and does not roundtrip with
/// [`TQPack`](crate::TQPack).
pub(crate) fn unpack_vector(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    // Get the input TurboQuant array into a form that is easier to work with.
    let parsed = parse_storage(input, ctx)?;
    let metadata = parsed.metadata;
    if parsed.len == 0 {
        return build_empty_vector(metadata, parsed.vector_validity);
    }

    let padded_dim = tq_padded_dim(metadata.dimensions)?;
    let transform = SorfMatrix::try_new(padded_dim, metadata.num_rounds as usize, metadata.seed)?;
    let padded_dim = u32::try_from(padded_dim)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;

    // We retrieve the centroids on read because they are mostly known statically for the given
    // settings.
    let centroids = compute_or_get_centroids(padded_dim, metadata.bit_width)?;

    match_each_float_ptype!(metadata.element_ptype, |T| {
        unpack_typed::<T>(
            DecodeInputs {
                metadata: &metadata,
                sorf_matrix: &transform,
                centroids: &centroids,
                norms: &parsed.norms,
                codes: &parsed.codes,
            },
            parsed.vector_validity,
            parsed.len,
            ctx,
        )
    })
}

fn build_empty_vector(
    metadata: TurboQuantMetadata,
    vector_validity: Validity,
) -> VortexResult<ArrayRef> {
    match_each_float_ptype!(metadata.element_ptype, |T| {
        let elements = PrimitiveArray::empty::<T>(Nullability::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            metadata.dimensions,
            vector_validity,
            0,
        )?;

        Vector::try_new_vector_array(fsl.into_array())
    })
}

struct DecodeInputs<'a> {
    metadata: &'a TurboQuantMetadata,
    sorf_matrix: &'a SorfMatrix,
    centroids: &'a [f32],
    norms: &'a PrimitiveArray,
    codes: &'a PrimitiveArray,
}

fn unpack_typed<T>(
    decode: DecodeInputs<'_>,
    vector_validity: Validity,
    num_vectors: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + Float + FromPrimitive,
{
    let metadata = decode.metadata;
    let dimensions = usize::try_from(metadata.dimensions)
        .vortex_expect("dimensions stays representable as usize");
    let padded_dim = decode.sorf_matrix.padded_dim();
    let centroids = decode.centroids;
    let norms = decode.norms.as_slice::<T>();
    let codes = decode.codes.as_slice::<u8>();
    let mask = vector_validity.execute_mask(num_vectors, ctx)?;

    let output_len = num_vectors
        .checked_mul(dimensions)
        .ok_or_else(|| vortex_err!("TurboQuant decoded vector length overflow"))?;
    let mut output = BufferMut::<T>::with_capacity(output_len);

    let mut decoded = vec![0.0f32; padded_dim];
    let mut inverse = vec![0.0f32; padded_dim];

    // Decode a single row: gather codes through the centroid table, apply the inverse SORF
    // transform, then denormalize and push `dimensions` elements into `output`. Captures the
    // read-only inputs and the scratch buffers so each call site only needs to pass `output`
    // and the row index.
    let mut decode_row = |output: &mut BufferMut<T>, i: usize| {
        let code_row = &codes[i * padded_dim..][..padded_dim];

        // Gather the values according to the codes.
        for (dst, &code) in decoded.iter_mut().zip(code_row.iter()) {
            *dst = *centroids
                .get(usize::from(code))
                .vortex_expect("TurboQuant code exceeds centroid count");
        }

        decode.sorf_matrix.inverse_transform(&decoded, &mut inverse);

        let norm = norms[i];
        for &value in inverse.iter().take(dimensions) {
            // `T::from_f32` is infallible for the supported float ptypes (`f16`, `f32`,
            // `f64`): values outside `f16` range saturate to `±inf` rather than returning
            // `None`.
            let value = T::from_f32(value)
                .vortex_expect("from_f32 is infallible for supported float types");

            // SAFETY: total pushes across all match arms equal `output_len`.
            unsafe { output.push_unchecked(value * norm) };
        }
    };

    // The total number of pushes is always exactly `num_vectors * dimensions == output_len`
    // across every arm below, which is the invariant the per-row `unsafe` blocks rely on.
    match &mask {
        Mask::AllFalse(_) => {
            // Every row is invalid: bulk-fill the output with zero placeholders.
            //
            // SAFETY: `output` was allocated with capacity `output_len`, and this push writes
            // exactly `output_len` zero placeholders.
            unsafe { output.push_n_unchecked(T::zero(), output_len) };
        }
        Mask::AllTrue(_) => {
            for i in 0..num_vectors {
                decode_row(&mut output, i);
            }
        }
        Mask::Values(values_mask) => {
            let mut cursor = 0;

            for &(start, end) in values_mask.slices() {
                if start > cursor {
                    // SAFETY: total pushes across all arms equal `output_len`.
                    unsafe { output.push_n_unchecked(T::zero(), (start - cursor) * dimensions) };
                }

                for i in start..end {
                    decode_row(&mut output, i);
                }

                cursor = end;
            }

            if cursor < num_vectors {
                // SAFETY: total pushes across all arms equal `output_len`.
                unsafe { output.push_n_unchecked(T::zero(), (num_vectors - cursor) * dimensions) };
            }
        }
    }

    let elements = PrimitiveArray::new::<T>(output.freeze(), Validity::NonNullable);
    let fsl = FixedSizeListArray::try_new(
        elements.into_array(),
        metadata.dimensions,
        vector_validity,
        num_vectors,
    )?;

    Vector::try_new_vector_array(fsl.into_array())
}
