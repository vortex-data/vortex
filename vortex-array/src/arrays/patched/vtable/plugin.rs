// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedSlots;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::patches::Patches;
use crate::patches::PatchesMetadata;
use crate::search_sorted::decode_sorted_positions;

/// The first four children are base, indices, patch values and chunk offsets. Replacement
/// reads patch values using the full outer dtype, including any dtype parameters. A future
/// UTF-8 or decimal replacement can therefore reuse the same slots.
///
/// Append a oneof tag for each new dtype family and allocate new operation numbers within
/// that family. Readers must reject unknown strategies: a required operand or a parameter
/// that changes the result belongs to a new operation. Trailing children of a known
/// strategy may be ignored, so they can carry optional indexes or statistics.
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

/// Check every chunk boundary, including the skipped patches in the first physical chunk.
/// Scalar lookups, slices and takes use this index while block execution streams the rows.
fn validate_chunk_index(
    indices: &ArrayRef,
    offsets: &ArrayRef,
    metadata: &PatchesMetadata,
    end: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let rows = decode_sorted_positions(indices, ctx)?;
    let offsets = decode_sorted_positions(offsets, ctx)?;
    let start = metadata.offset()?;
    vortex_ensure!(rows.is_sorted_by(|a, b| a < b));
    vortex_ensure!(rows[0] >= start as u64 && rows[rows.len() - 1] < end as u64);
    let skipped = metadata.offset_within_chunk();
    let mut rows = rows.iter().copied().peekable();
    let mut consumed = 0u64;
    for (chunk, &offset) in offsets.iter().enumerate() {
        if chunk != 0 {
            let expected = offsets[0]
                .checked_add(skipped)
                .and_then(|base| base.checked_add(consumed))
                .ok_or_else(|| vortex_err!("Patch chunk offset overflow"))?;
            vortex_ensure_eq!(
                offset,
                expected,
                "Chunk offset disagrees with patch indices"
            );
        }
        let physical_chunk = start / PATCH_CHUNK_SIZE + chunk;
        while rows
            .next_if(|row| *row / PATCH_CHUNK_SIZE as u64 == physical_chunk as u64)
            .is_some()
        {
            consumed += 1;
        }
    }
    Ok(())
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
        vortex_ensure!(parts.children.len() >= PatchedSlots::COUNT);
        let metadata = PatchedMetadata::decode(parts.metadata)?;
        let strategy = match metadata
            .strategy
            .ok_or_else(|| vortex_err!("Missing patch strategy"))?
        {
            Strategy::Integer(op) => PatchStrategy::Integer(
                IntegerPatchStrategy::try_from(op)
                    .map_err(|_| vortex_err!("Unknown integer patch strategy {op}"))?,
            ),
            Strategy::Float(op) => PatchStrategy::Float(
                FloatPatchStrategy::try_from(op)
                    .map_err(|_| vortex_err!("Unknown float patch strategy {op}"))?,
            ),
        };
        strategy.validate(parts.dtype)?;
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
        let indices = children.get(
            PatchedSlots::PATCH_INDICES,
            &metadata.indices_dtype()?,
            count,
        )?;
        let values = children.get(PatchedSlots::PATCH_VALUES, parts.dtype, count)?;
        let offsets = children.get(PatchedSlots::PATCH_CHUNK_OFFSETS, &offsets_dtype, chunks)?;
        let mut ctx = session.create_execution_ctx();
        validate_chunk_index(&indices, &offsets, &metadata, end, &mut ctx)?;
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

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::PatchedMetadata;
    use super::Strategy;
    use crate::ArrayDeserialization;
    use crate::ArrayPlugin;
    use crate::IntoArray;
    use crate::arrays::Patched;
    use crate::dtype::PType;
    use crate::patches::PatchesMetadata;
    use crate::test_harness::check_metadata;

    #[test]
    fn float_metadata_golden() {
        let metadata = PatchedMetadata {
            patches: Some(PatchesMetadata::new(
                3,
                1025,
                PType::U16,
                Some(2),
                Some(PType::U16),
                Some(1),
            )),
            strategy: Some(Strategy::Float(0)),
        };
        check_metadata("patched.metadata", &metadata.encode_to_vec());
    }

    #[test]
    fn reject_invalid_wire_metadata() {
        let session = VortexSession::empty();
        let children = [
            buffer![0u8; 2050].into_array(),
            buffer![1u16, 1024].into_array(),
            buffer![2u8, 3].into_array(),
            buffer![0u8, 1, 2].into_array(),
        ];
        let patches = |count, chunks, dtype| {
            PatchesMetadata::new(count, 0, PType::U16, Some(chunks), dtype, Some(0))
        };
        let valid = patches(2, 3, Some(PType::U8));
        let decode = |bytes: Vec<u8>, children: &[crate::ArrayRef]| {
            let parts = ArrayDeserialization::new(
                ArrayPlugin::id(&Patched),
                children[0].dtype(),
                2050,
                &bytes,
                &[],
                &children,
            );
            ArrayPlugin::deserialize(&Patched, parts, &session)
        };
        let integer = |patches| {
            PatchedMetadata {
                patches,
                strategy: Some(Strategy::Integer(0)),
            }
            .encode_to_vec()
        };
        assert!(decode(integer(Some(valid)), &children).is_ok());
        let reject = |bytes| assert!(decode(bytes, &children).is_err());
        for metadata in [
            None,
            Some(patches(0, 3, Some(PType::U8))),
            Some(patches(2051, 3, Some(PType::U8))),
            Some(patches(2, 3, None)),
            Some(patches(2, 2, Some(PType::U8))),
        ] {
            reject(integer(metadata));
        }
        reject(
            PatchedMetadata {
                patches: Some(valid),
                strategy: Some(Strategy::Float(0)),
            }
            .encode_to_vec(),
        );
        let mut future_family = PatchedMetadata {
            patches: Some(valid),
            strategy: None,
        }
        .encode_to_vec();
        future_family.extend_from_slice(&[0x20, 0]); // field 4: a future family's Replace
        reject(future_family);
        let mut inconsistent = children;
        inconsistent[3] = buffer![0u8, 1, 1].into_array();
        assert!(decode(integer(Some(valid)), &inconsistent).is_err());
    }
}
