// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::FloatPatchStrategy;
use super::IntegerPatchStrategy;
use super::PatchStrategy;
use super::Patched;
use super::PatchedArrayExt;
use super::PatchedSlots;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::patches::PatchesMetadata;

/// Wire metadata for the sorted layout. All four children are required; there are no buffers.
#[derive(Clone, prost::Message)]
struct PatchedMetadata {
    #[prost(message, optional, tag = "1")]
    patches: Option<PatchesMetadata>,
    #[prost(oneof = "Strategy", tags = "2, 3")]
    strategy: Option<Strategy>,
}

#[derive(Clone, prost::Oneof)]
enum Strategy {
    #[prost(enumeration = "IntegerPatchStrategy", tag = "2")]
    Integer(i32),
    #[prost(enumeration = "FloatPatchStrategy", tag = "3")]
    Float(i32),
}

impl ArrayPlugin for Patched {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<Self>()
            .ok_or_else(|| vortex_err!("Expected Patched array"))?;
        let strategy = match view.strategy() {
            PatchStrategy::Integer(op) => Strategy::Integer(op as i32),
            PatchStrategy::Float(op) => Strategy::Float(op as i32),
        };
        let metadata = PatchedMetadata {
            patches: Some(view.patches().to_metadata(array.len(), array.dtype())?),
            strategy: Some(strategy),
        };
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(parts.serialized_id, VTable::id(self));
        vortex_ensure!(parts.buffers.is_empty());
        vortex_ensure_eq!(parts.children.len(), PatchedSlots::COUNT);
        let metadata = PatchedMetadata::decode(parts.metadata)?;
        let strategy = match metadata
            .strategy
            .ok_or_else(|| vortex_err!("Missing patch strategy"))?
        {
            Strategy::Integer(op) => PatchStrategy::Integer(IntegerPatchStrategy::try_from(op)
                .map_err(|_| vortex_err!("Unknown integer patch strategy {op}"))?),
            Strategy::Float(op) => PatchStrategy::Float(FloatPatchStrategy::try_from(op)
                .map_err(|_| vortex_err!("Unknown float patch strategy {op}"))?),
        };
        vortex_ensure!(strategy == PatchStrategy::replace(parts.dtype)?);
        let metadata = metadata
            .patches
            .ok_or_else(|| vortex_err!("Missing patches metadata"))?;
        let count = metadata.len()?;
        vortex_ensure!((1..=parts.len).contains(&count));
        let offsets_dtype = metadata
            .chunk_offsets_dtype()?
            .ok_or_else(|| vortex_err!("Missing patch chunk offsets dtype"))?;
        vortex_ensure!(offsets_dtype.is_unsigned_int());
        let offset = metadata.offset()?;
        let end = offset
            .checked_add(parts.len)
            .ok_or_else(|| vortex_err!("Patch range overflow"))?;
        let chunks = end.div_ceil(PATCH_CHUNK_SIZE) - offset / PATCH_CHUNK_SIZE;
        vortex_ensure_eq!(usize::try_from(metadata.chunk_offsets_len())?, chunks);
        let children = parts.children;
        let inner = children.get(PatchedSlots::INNER, parts.dtype, parts.len)?;
        let indices = children.get(PatchedSlots::PATCH_INDICES, &metadata.indices_dtype()?, count)?;
        let values = children.get(PatchedSlots::PATCH_VALUES, parts.dtype, count)?;
        let offsets = children.get(PatchedSlots::PATCH_CHUNK_OFFSETS, &offsets_dtype, chunks)?;
        let mut ctx = session.create_execution_ctx();
        let patches = Patches::from_metadata(
            parts.len,
            &metadata,
            indices,
            values,
            Some(offsets),
            &mut ctx,
        )?;
        Ok(Self::with_strategy(inner, &patches, strategy, &mut ctx)?.into_array())
    }
}
