// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
use vortex_layout::layouts::compressed::CompressorPlugin;
use vortex_zstd::ZstdArray;

const GPU_ZSTD_LEVEL: i32 = 3;
const GPU_ZSTD_VALUES_PER_PAGE: usize = 8192;

#[derive(Clone)]
pub(super) struct GpuCompatibleCompressor {
    btrblocks: BtrBlocksCompressor,
}

impl GpuCompatibleCompressor {
    pub(super) const fn new(btrblocks: BtrBlocksCompressor) -> Self {
        Self { btrblocks }
    }

    fn compress_canonical(&self, canonical: Canonical) -> VortexResult<ArrayRef> {
        match canonical {
            // Use nvcomp-compatible zstd (without dictionary) for string/binary leaves.
            Canonical::VarBinView(vbv) => {
                let zstd = ZstdArray::from_var_bin_view_without_dict(
                    &vbv,
                    GPU_ZSTD_LEVEL,
                    GPU_ZSTD_VALUES_PER_PAGE,
                )?
                .into_array();
                if zstd.nbytes() < vbv.nbytes() {
                    Ok(zstd)
                } else {
                    Ok(vbv.into_array())
                }
            }
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .unmasked_fields()
                    .iter()
                    .map(|field| self.compress_canonical(field.to_canonical()?))
                    .collect::<VortexResult<Vec<_>>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity()?,
                )?
                .into_array())
            }
            Canonical::List(list_view) => {
                let compressed_elems =
                    self.compress_canonical(list_view.elements().to_canonical()?)?;
                let compressed_offsets =
                    self.compress_canonical(list_view.offsets().to_canonical()?)?;
                let compressed_sizes =
                    self.compress_canonical(list_view.sizes().to_canonical()?)?;

                Ok(ListViewArray::try_new(
                    compressed_elems,
                    compressed_offsets,
                    compressed_sizes,
                    list_view.validity()?,
                )?
                .into_array())
            }
            Canonical::FixedSizeList(fsl) => {
                let compressed_elems = self.compress_canonical(fsl.elements().to_canonical()?)?;
                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl.list_size(),
                    fsl.validity()?,
                    fsl.len(),
                )?
                .into_array())
            }
            Canonical::Extension(ext) => {
                let compressed_storage = self.compress_canonical(ext.storage().to_canonical()?)?;
                Ok(ExtensionArray::new(ext.ext_dtype().clone(), compressed_storage).into_array())
            }
            other => self.btrblocks.compress(other.as_ref()),
        }
    }
}

impl CompressorPlugin for GpuCompatibleCompressor {
    fn compress_chunk(&self, chunk: &dyn Array) -> VortexResult<ArrayRef> {
        self.compress_canonical(chunk.to_canonical()?)
    }
}
