// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.for.v2`, which stores one reference per chunk as a child.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::for_v2_id;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;
use crate::for_::array::num_chunks;

/// Metadata for `fastlanes.for.v2`.
///
/// The children are `[encoded, references]`, with one non-nullable reference per 1024-element
/// chunk spanned by `offset + len` elements. Neither child is optional.
///
/// Readers require at least these two children and ignore any that follow. A later writer may
/// append a child under this ID only if decoding without it still yields the same values, such as
/// a statistic or an index. A child or metadata field that changes the decoded values needs a new
/// encoding ID.
#[derive(Clone, prost::Message)]
pub(super) struct FoRV2Metadata {
    /// The position of the first element within the first chunk.
    #[prost(uint32, tag = "1")]
    pub(super) offset: u32,
}

pub(super) fn serialize(array: ArrayView<'_, FoR>, references: &ArrayRef) -> ArraySerialization {
    let metadata = FoRV2Metadata {
        offset: u32::from(array.offset()),
    };
    ArraySerialization::new(
        for_v2_id(),
        metadata.encode_to_vec(),
        vec![],
        vec![array.encoded().clone(), references.clone()],
    )
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        parts.buffers.is_empty(),
        "FoRArray expects 0 buffers, got {}",
        parts.buffers.len()
    );
    vortex_ensure!(
        parts.children.len() >= 2,
        "Expected at least 2 children for {}, got {}",
        for_v2_id(),
        parts.children.len()
    );
    let metadata = FoRV2Metadata::decode(parts.metadata)?;
    vortex_ensure!(
        parts.dtype.is_int(),
        "FoR requires an integer dtype, got {}",
        parts.dtype
    );
    vortex_ensure!(
        usize::try_from(metadata.offset).is_ok_and(|offset| offset < FL_CHUNK_SIZE),
        "FoR offset must be less than {FL_CHUNK_SIZE}, got {}",
        metadata.offset
    );
    let offset = u16::try_from(metadata.offset)?;
    vortex_ensure!(
        usize::from(offset).checked_add(parts.len).is_some(),
        "FoR offset {offset} + len {} overflowed usize",
        parts.len
    );
    let encoded = parts.children.get(0, parts.dtype, parts.len)?;
    let references = parts.children.get(
        1,
        &parts.dtype.as_nonnullable(),
        num_chunks(offset, parts.len),
    )?;
    Ok(FoR::try_new_chunked(encoded, references, offset)?.into_array())
}
