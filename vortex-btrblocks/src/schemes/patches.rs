// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::patches::Patches;
use vortex_error::VortexError;
use vortex_error::VortexResult;

/// Compresses the given patches by downscaling integers and checking for constant values.
pub fn compress_patches(patches: Patches) -> VortexResult<Patches> {
    // Downscale the patch indices.
    let indices = patches.indices().to_primitive().narrow()?.into_array();

    // Check if the values are constant.
    let values = patches.values();
    let values = if values
        .statistics()
        .compute_is_constant()
        .unwrap_or_default()
    {
        ConstantArray::new(values.scalar_at(0)?, values.len()).into_array()
    } else {
        values.clone()
    };
    let chunk_offsets = patches
        .chunk_offsets()
        .as_ref()
        .map(|offsets| Ok::<ArrayRef, VortexError>(offsets.to_primitive().narrow()?.into_array()))
        .transpose()?;

    Patches::new(
        patches.array_len(),
        patches.offset(),
        indices,
        values,
        chunk_offsets,
    )
}
