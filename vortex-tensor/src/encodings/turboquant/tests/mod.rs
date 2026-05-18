// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for TurboQuant encoding with decomposed SorfTransform + DictArray tree.

mod compute;
mod nullable;
mod roundtrip;
mod structural;

use std::f32;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::encodings::turboquant::TurboQuantConfig;
use crate::encodings::turboquant::turboquant_encode;
use crate::tests::SESSION;
use crate::types::vector::Vector;

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
fn make_vector_ext(fsl: &FixedSizeListArray) -> ArrayRef {
    Vector::try_new_vector_array(fsl.clone().into_array())
        .vortex_expect("test FSL satisfies Vector storage constraints")
}

/// Unwrap an L2Denorm ScalarFnArray into (sorf_child, norms_child).
fn unwrap_l2denorm(encoded: &ArrayRef) -> (ArrayRef, ArrayRef) {
    let sfn = encoded
        .as_opt::<ScalarFn>()
        .expect("expected ScalarFnArray (L2Denorm)");
    (sfn.child_at(0).clone(), sfn.child_at(1).clone())
}

/// Navigate the full tree to get (codes, centroids, norms) as flat arrays.
fn unwrap_codes_centroids_norms(
    encoded: &ArrayRef,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray, PrimitiveArray)> {
    let (sorf_child, norms_child) = unwrap_l2denorm(encoded);
    let padded_vector_child = sorf_child
        .as_opt::<ScalarFn>()
        .expect("expected SorfTransform ScalarFnArray")
        .child_at(0)
        .clone();

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
    let sqrt3_pi_over_2 = (3.0f32).sqrt() * f32::consts::PI / 2.0;
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
    let encoded = turboquant_encode(make_vector_ext(fsl), config, &mut ctx)?;
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
