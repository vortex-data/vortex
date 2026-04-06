// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant decoding (dequantization) logic.

use num_traits::Float;
use num_traits::FromPrimitive;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_float_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::array::float_from_f32;
use crate::encodings::turboquant::array::rotation::RotationMatrix;
use crate::utils::tensor_element_ptype;

/// Decompress a `TurboQuantArray` into a [`Vector`] extension array.
///
/// The returned array is an [`ExtensionArray`] with the original Vector dtype wrapping a
/// `FixedSizeListArray` of f32 elements.
///
/// [`Vector`]: crate::vector::Vector
pub fn execute_decompress(
    array: Array<TurboQuant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dim = array.dimension() as usize;
    let padded_dim = array.padded_dim() as usize;
    let num_rows = array.norms().len();
    let ext_dtype = array.dtype().as_extension().clone();
    let element_ptype = tensor_element_ptype(&ext_dtype)?;

    if num_rows == 0 {
        let fsl_validity = Validity::from(ext_dtype.storage_dtype().nullability());

        match_each_float_ptype!(element_ptype, |T| {
            let elements = PrimitiveArray::empty::<T>(Nullability::NonNullable);
            let fsl = FixedSizeListArray::try_new(
                elements.into_array(),
                array.dimension(),
                fsl_validity,
                0,
            )?;

            return Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array());
        })
    }

    // Read stored centroids (always f32).
    let centroids_prim = array.centroids().clone().execute::<PrimitiveArray>(ctx)?;
    let centroids = centroids_prim.as_slice::<f32>();

    // FastLanes SIMD-unpacks the 1-bit bitpacked rotation signs into u8 0/1 values,
    // then we expand to u32 XOR masks once (amortized over all rows). This enables
    // branchless XOR-based sign application in the per-row SRHT hot loop.
    let signs_prim = array
        .rotation_signs()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let rotation = RotationMatrix::from_u8_slice(signs_prim.as_slice::<u8>(), dim)?;

    // Unpack codes from FixedSizeListArray -> flat u8 elements.
    let codes_fsl = array.codes().clone().execute::<FixedSizeListArray>(ctx)?;
    let codes_prim = codes_fsl.elements().to_canonical()?.into_primitive();
    let indices = codes_prim.as_slice::<u8>();

    // Read norms in their native precision. Norms carry the validity of the array.
    let norms_prim = array.norms().clone().execute::<PrimitiveArray>(ctx)?;
    let output_validity = array.norms().validity()?;

    // MSE decode: dequantize (f32) -> inverse rotate (f32) -> scale by norm -> cast to T.
    // The rotation and centroid lookup always happen in f32. The final output is cast to the
    // Vector's element type to match the original storage dtype.
    match_each_float_ptype!(element_ptype, |T| {
        decompress_typed::<T>(
            &norms_prim,
            centroids,
            &rotation,
            indices,
            dim,
            padded_dim,
            num_rows,
        )
        .and_then(|elements| {
            let fsl = FixedSizeListArray::try_new(
                elements.into_array(),
                array.dimension(),
                output_validity,
                num_rows,
            )?;
            Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
        })
    })
}

/// Typed decompress: reads norms as `T`, dequantizes in f32, and produces output as `T`.
fn decompress_typed<T: NativePType + Float + FromPrimitive>(
    norms_prim: &PrimitiveArray,
    centroids: &[f32],
    rotation: &RotationMatrix,
    indices: &[u8],
    dim: usize,
    padded_dim: usize,
    num_rows: usize,
) -> VortexResult<PrimitiveArray> {
    let norms = norms_prim.as_slice::<T>();

    let mut output = BufferMut::<T>::with_capacity(num_rows * dim);
    let mut dequantized = vec![0.0f32; padded_dim];
    let mut unrotated = vec![0.0f32; padded_dim];

    for row in 0..num_rows {
        let row_indices = &indices[row * padded_dim..(row + 1) * padded_dim];
        let norm = norms[row];

        for idx in 0..padded_dim {
            dequantized[idx] = centroids[row_indices[idx] as usize];
        }

        rotation.inverse_rotate(&dequantized, &mut unrotated);

        for idx in 0..dim {
            let val = float_from_f32::<T>(unrotated[idx]) * norm;
            output.push(val);
        }
    }

    Ok(PrimitiveArray::new::<T>(
        output.freeze(),
        Validity::NonNullable,
    ))
}
