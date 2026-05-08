// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant packing (quantization) logic.
//!
//! The input to [`pack_vector()`] must be a [`Vector`](vortex_tensor::vector::Vector) extension
//! array. [`pack_vector()`] computes original row norms, normalizes valid rows internally via
//! [`tq_normalize_as_l2_denorm()`], quantizes the normalized child, and stores row-aligned norms and
//! codes in the TurboQuant extension storage.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_tensor::vector::AnyVector;

use super::normalize::tq_normalize_as_l2_denorm;
use super::quantize::empty_quantization;
use super::quantize::turboquant_quantize_core;
use super::storage::build_codes_child;
use super::storage::build_storage;
use super::tq_padded_dim;
use crate::TurboQuantConfig;
use crate::config::MIN_DIMENSION;
use crate::vtable::TurboQuant;
use crate::vtable::TurboQuantMetadata;

/// Lossily pack a `Vector` extension array into a `TurboQuant` extension array.
///
/// Valid rows are normalized internally before SORF transform and scalar quantization. The original
/// row norms are stored explicitly, and original vector nulls are preserved on the storage struct
/// and both row-aligned child arrays.
pub(crate) fn pack_vector(
    input: ArrayRef,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let num_vectors = input.len();
    let vector_metadata = input
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
        .ok_or_else(|| vortex_err!("TurboQuant pack expects a Vector extension array"))?;

    let element_ptype = vector_metadata.element_ptype();

    let dimensions = vector_metadata.dimensions();
    vortex_ensure!(
        dimensions >= MIN_DIMENSION,
        "TurboQuant requires dimension >= {MIN_DIMENSION}, got {dimensions}",
    );
    let padded_dim = tq_padded_dim(dimensions)?;

    let vector_validity = input.validity()?;

    // We must normalize the vectors in order to apply the transform during quantization.
    // NB: The 2 child arrays share the same validity with `input`.
    let l2_denorm = tq_normalize_as_l2_denorm(input, ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();

    let normalized_ext = normalized
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_err!("normalized TurboQuant input must be a Vector extension"))?;
    let normalized_fsl: FixedSizeListArray = normalized_ext.storage_array().clone().execute(ctx)?;

    let core = if normalized_fsl.is_empty() {
        empty_quantization(padded_dim)
    } else {
        // SAFETY: `tq_normalize_as_l2_denorm` returned this normalized Vector child.
        unsafe { turboquant_quantize_core(&normalized_fsl, config, ctx)? }
    };
    let codes = build_codes_child(num_vectors, core, vector_validity.clone())?;

    // Now that we have the codes into the centroid codebook and the norms, we can build the
    // TurboQuant extension array.
    let metadata = TurboQuantMetadata {
        element_ptype,
        dimensions,
        bit_width: config.bit_width(),
        seed: config.seed(),
        num_rounds: config.num_rounds(),
    };
    let storage = build_storage(norms, codes, num_vectors, vector_validity)?;

    Ok(ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage)?.into_array())
}
