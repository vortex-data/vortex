// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressor plugin that applies TurboQuant to tensor extension columns.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_layout::layouts::compressed::CompressorPlugin;

use crate::TurboQuantConfig;
use crate::array::TurboQuantVariant;
use crate::compress::turboquant_encode;

/// Extension IDs for tensor types (from vortex-tensor).
const VECTOR_EXT_ID: &str = "vortex.tensor.vector";
const FIXED_SHAPE_TENSOR_EXT_ID: &str = "vortex.tensor.fixed_shape_tensor";

/// A [`CompressorPlugin`] that applies TurboQuant to Vector and FixedShapeTensor
/// extension columns, and delegates all other compression to an inner plugin.
///
/// After TurboQuant encoding, each child of the resulting `TurboQuantArray` is
/// recursively compressed by the inner compressor so that norms, codes, etc.
/// benefit from the normal compression strategy.
pub struct TurboQuantCompressor {
    config: TurboQuantConfig,
    inner: Arc<dyn CompressorPlugin>,
}

impl TurboQuantCompressor {
    /// Create a new compressor that wraps an inner compressor.
    pub fn new(config: TurboQuantConfig, inner: Arc<dyn CompressorPlugin>) -> Self {
        Self { config, inner }
    }
}

/// Check if an extension array has a tensor extension type.
fn is_tensor_extension(ext_array: &ExtensionArray) -> bool {
    let ext_id = ext_array.ext_dtype().id();
    ext_id.as_ref() == VECTOR_EXT_ID || ext_id.as_ref() == FIXED_SHAPE_TENSOR_EXT_ID
}

impl CompressorPlugin for TurboQuantCompressor {
    fn compress_chunk(&self, chunk: &ArrayRef) -> VortexResult<ArrayRef> {
        let canonical = chunk.to_canonical()?;
        if let Canonical::Extension(ext_array) = &canonical
            && is_tensor_extension(ext_array)
        {
            return self.compress_tensor(ext_array);
        }

        self.inner.compress_chunk(chunk)
    }
}

impl TurboQuantCompressor {
    fn compress_tensor(&self, ext_array: &ExtensionArray) -> VortexResult<ArrayRef> {
        let storage = ext_array.storage_array();
        let fsl = storage.to_canonical()?.into_fixed_size_list();
        let tq_array = turboquant_encode(&fsl, &self.config)?;

        // Recursively compress each child via the inner compressor.
        let compressed_codes = self.inner.compress_chunk(tq_array.codes())?;
        let compressed_norms = self.inner.compress_chunk(tq_array.norms())?;

        let compressed_tq = match tq_array.variant() {
            TurboQuantVariant::Mse => crate::TurboQuantArray::try_new_mse(
                fsl.dtype().clone(),
                compressed_codes,
                compressed_norms,
                tq_array.dimension(),
                tq_array.bit_width(),
                tq_array.rotation_seed(),
            )?,
            TurboQuantVariant::Prod => {
                let compressed_qjl = self.inner.compress_chunk(
                    tq_array
                        .qjl_signs()
                        .vortex_expect("Prod variant must have qjl_signs"),
                )?;
                let compressed_res_norms = self.inner.compress_chunk(
                    tq_array
                        .residual_norms()
                        .vortex_expect("Prod variant must have residual_norms"),
                )?;

                crate::TurboQuantArray::try_new_prod(
                    fsl.dtype().clone(),
                    compressed_codes,
                    compressed_norms,
                    compressed_qjl,
                    compressed_res_norms,
                    tq_array.dimension(),
                    tq_array.bit_width(),
                    tq_array.rotation_seed(),
                )?
            }
        };

        Ok(
            ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_tq.into_array())
                .into_array(),
        )
    }
}
