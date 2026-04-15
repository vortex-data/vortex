// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for TurboQuant encoding with decomposed SorfTransform + DictArray tree.

mod compute;
mod nullable;
mod roundtrip;
mod structural;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnVTable;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::extension::EmptyMetadata;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::encodings::turboquant::TurboQuantConfig;
use crate::encodings::turboquant::turboquant_encode_unchecked;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;
use crate::tests::SESSION;
use crate::vector::Vector;

/// Create a FixedSizeListArray of random f32 vectors with the given validity.
fn make_fsl_with_validity(
    num_rows: usize,
    dim: usize,
    seed: u64,
    validity: Validity,
) -> FixedSizeListArray {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim);
    for _ in 0..(num_rows * dim) {
        buf.push(normal.sample(&mut rng));
    }

    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        validity,
        num_rows,
    )
    .unwrap()
}

/// Create a non-nullable FixedSizeListArray of random f32 vectors.
fn make_fsl(num_rows: usize, dim: usize, seed: u64) -> FixedSizeListArray {
    make_fsl_with_validity(num_rows, dim, seed, Validity::NonNullable)
}

/// Wrap a `FixedSizeListArray` in a `Vector` extension array.
fn make_vector_ext(fsl: &FixedSizeListArray) -> ExtensionArray {
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())
        .unwrap()
        .erased();
    ExtensionArray::new(ext_dtype, fsl.clone().into_array())
}

/// Full encode pipeline: normalize → TQ-encode → wrap in L2Denorm.
fn normalize_and_encode(
    ext: &ExtensionArray,
    config: &TurboQuantConfig,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let l2_denorm = normalize_as_l2_denorm(ext.as_ref().clone(), ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();
    let num_rows = l2_denorm.len();

    let normalized_ext = normalized
        .as_opt::<Extension>()
        .vortex_expect("normalized child should be an Extension array");
    // SAFETY: We just normalized the input via `normalize_as_l2_denorm`.
    let tq = unsafe { turboquant_encode_unchecked(normalized_ext, config, ctx)? };

    Ok(unsafe { L2Denorm::new_array_unchecked(tq, norms, num_rows) }?.into_array())
}

/// Unwrap an L2Denorm ScalarFnArray into (sorf_child, norms_child).
fn unwrap_l2denorm(encoded: &ArrayRef) -> (ArrayRef, ArrayRef) {
    let sfn = encoded
        .as_opt::<ScalarFnVTable>()
        .expect("expected ScalarFnArray (L2Denorm)");
    let sorf_child = sfn.child_at(0).clone();
    let norms_child = sfn.child_at(1).clone();
    (sorf_child, norms_child)
}

/// Unwrap a SorfTransform ScalarFnArray to get the FSL(Dict) child.
fn unwrap_sorf(sorf: &ArrayRef) -> ArrayRef {
    let sfn = sorf
        .as_opt::<ScalarFnVTable>()
        .expect("expected ScalarFnArray (SorfTransform)");
    sfn.child_at(0).clone()
}

/// Navigate the full tree to get (codes, centroids, norms) as flat arrays.
fn unwrap_codes_centroids_norms(
    encoded: &ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray, PrimitiveArray)> {
    let (sorf_child, norms_child) = unwrap_l2denorm(encoded);
    let padded_vector_child = unwrap_sorf(&sorf_child);

    // Vector<padded_dim> wrapping FSL(Dict(codes, centroids))
    let padded_vector: ExtensionArray = padded_vector_child.execute(ctx)?;
    let fsl: FixedSizeListArray = padded_vector.storage_array().clone().execute(ctx)?;
    let dict = fsl
        .elements()
        .as_opt::<Dict>()
        .vortex_expect("FSL elements should be a DictArray");
    let codes: PrimitiveArray = dict.codes().clone().execute(ctx)?;
    let centroids: PrimitiveArray = dict.values().clone().execute(ctx)?;
    let norms: PrimitiveArray = norms_child.execute(ctx)?;

    Ok((codes, centroids, norms))
}

fn theoretical_mse_bound(bit_width: u8) -> f32 {
    let sqrt3_pi_over_2 = (3.0f32).sqrt() * std::f32::consts::PI / 2.0;
    sqrt3_pi_over_2 / (4.0f32).powi(bit_width as i32)
}

fn per_vector_normalized_mse(
    original: &[f32],
    reconstructed: &[f32],
    dim: usize,
    num_rows: usize,
) -> f32 {
    let mut total = 0.0f32;
    for row in 0..num_rows {
        let orig = &original[row * dim..(row + 1) * dim];
        let recon = &reconstructed[row * dim..(row + 1) * dim];
        let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
        if norm_sq < 1e-10 {
            continue;
        }
        let err_sq: f32 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        total += err_sq / norm_sq;
    }
    total / num_rows as f32
}

/// Normalize, encode, and decode, returning (original, decoded) flat f32 slices.
fn encode_decode(
    fsl: &FixedSizeListArray,
    config: &TurboQuantConfig,
) -> VortexResult<(Vec<f32>, Vec<f32>)> {
    let mut ctx = SESSION.create_execution_ctx();
    let original: Vec<f32> = {
        let prim = fsl.elements().clone().execute::<PrimitiveArray>(&mut ctx)?;
        prim.as_slice::<f32>().to_vec()
    };
    let ext = make_vector_ext(fsl);
    let encoded = normalize_and_encode(&ext, config, &mut ctx)?;
    let decoded_ext = encoded.execute::<ExtensionArray>(&mut ctx)?;
    let decoded_fsl = decoded_ext
        .storage_array()
        .clone()
        .execute::<FixedSizeListArray>(&mut ctx)?;
    let decoded_elements: Vec<f32> = {
        let prim = decoded_fsl
            .elements()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        prim.as_slice::<f32>().to_vec()
    };
    Ok((original, decoded_elements))
}

fn make_fsl_small(dim: usize) -> FixedSizeListArray {
    let mut buf = BufferMut::<f32>::with_capacity(dim);
    for i in 0..dim {
        buf.push(i as f32 + 1.0);
    }
    let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
    FixedSizeListArray::try_new(
        elements.into_array(),
        dim.try_into()
            .expect("somehow got dimension greater than u32::MAX"),
        Validity::NonNullable,
        1,
    )
    .unwrap()
}
