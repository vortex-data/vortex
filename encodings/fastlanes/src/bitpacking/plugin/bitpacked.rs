// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-packed wire formats and their serialization plugin.

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArraySlots;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::ChunkLayout;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedSlots;
use crate::bitpacking::array::CHUNK_OFFSETS_DTYPE;

/// Metadata of the frozen `fastlanes.bitpacked` wire format.
#[derive(Clone, prost::Message)]
pub(crate) struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

/// Read the frozen v1 metadata, packed buffer, and patch/validity children.
fn deserialize_v1(parts: ArrayDeserialization<'_>) -> VortexResult<ArrayRef> {
    let ArrayDeserialization {
        dtype,
        len,
        metadata,
        buffers,
        children,
        ..
    } = parts;

    let metadata = BitPackedMetadata::decode(metadata)?;
    let packed = single_buffer(buffers)?;
    let (patches, validity, _) = deserialize_children(children, metadata.patches, dtype, len, 0)?;
    let bit_width = u8::try_from(metadata.bit_width).map_err(|_| {
        vortex_err!(
            "BitPackedMetadata bit_width {} does not fit in u8",
            metadata.bit_width
        )
    })?;
    vortex_ensure!(bit_width <= 64, "Unsupported bit width {bit_width}");
    let offset = offset_from_metadata(metadata.offset)?;
    let num_chunks = (len + offset as usize).div_ceil(FL_CHUNK_SIZE);

    let slots = {
        let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
        PatchesData::push_slots(&mut s, patches.as_ref());
        s.push(validity_to_child(&validity, len));
        let widths = ChunkLayout::uniform(bit_width, num_chunks);
        let offsets = widths.offsets_array();
        s.push(Some(offsets));
        s
    };
    let data = BitPackedData::try_new(packed, patches, offset)?;
    Ok(Array::<BitPacked>::try_from_parts(
        ArrayParts::new(BitPacked, dtype.clone(), len, data).with_slots(slots),
    )?
    .into_array())
}

/// The single packed buffer of a serialized bit-packed array.
fn single_buffer(buffers: &[BufferHandle]) -> VortexResult<BufferHandle> {
    vortex_ensure!(
        buffers.len() == 1,
        "Expected 1 buffer, got {}",
        buffers.len()
    );
    Ok(buffers[0].clone())
}

/// The offset into the first chunk, which the metadata stores as a `u32`.
fn offset_from_metadata(offset: u32) -> VortexResult<u16> {
    u16::try_from(offset)
        .map_err(|_| vortex_err!("BitPackedMetadata offset {offset} does not fit in u16"))
}

/// Read the patches and validity children that both wire formats share.
///
/// Children run: the patches, then a validity bitmap if there is one, then `trailing` children
/// the caller reads itself. Returns the index of the first trailing child.
fn deserialize_children(
    children: &dyn ArrayChildren,
    patches: Option<PatchesMetadata>,
    dtype: &DType,
    len: usize,
    trailing: usize,
) -> VortexResult<(Option<Patches>, Validity, usize)> {
    let num_patch_children = match &patches {
        None => 0,
        Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
        Some(_) => 2,
    };
    let num_fixed = num_patch_children + trailing;
    let has_validity = match children.len().checked_sub(num_fixed) {
        Some(0) => false,
        Some(1) => true,
        _ => vortex_bail!(
            "Expected {num_fixed} or {} children, got {}",
            num_fixed + 1,
            children.len()
        ),
    };
    let validity = if has_validity {
        Validity::Array(children.get(num_patch_children, &Validity::DTYPE, len)?)
    } else {
        Validity::from(dtype.nullability())
    };
    let patches = patches
        .map(|p| {
            let indices = children.get(0, &p.indices_dtype()?, p.len()?)?;
            let values = children.get(1, dtype, p.len()?)?;
            let chunk_offsets = p
                .chunk_offsets_dtype()?
                .map(|dtype| children.get(2, &dtype, p.chunk_offsets_len() as usize))
                .transpose()?;
            Patches::new(len, p.offset()?, indices, values, chunk_offsets)
        })
        .transpose()?;
    Ok((
        patches,
        validity,
        num_patch_children + usize::from(has_validity),
    ))
}

/// The serialized format for arrays whose chunks do not all share one bit width.
///
/// The original `fastlanes.bitpacked` format carries a single `bit_width`, and readers of that
/// format assume every chunk uses it. Arrays with differing chunk widths therefore serialize under
/// this successor ID, which older readers reject as unknown instead of misreading.
pub fn bitpacked_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.bitpacked_v2");
    *ID
}

/// Metadata of the `fastlanes.bitpacked_v2` format. Chunk byte boundaries travel
/// in one child, keeping the metadata bounded.
///
/// Tag 1 is left unused: it is `bit_width` in the original format, so metadata misdirected across
/// the two IDs decodes to the right fields and fails on the child layout instead.
#[derive(Clone, prost::Message)]
pub(crate) struct BitPackedV2Metadata {
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32,
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

/// The [`ArrayPlugin`] for `BitPacked`, owning both of its wire formats.
///
/// Arrays whose chunks share one width serialize without the offsets child as the
/// frozen `fastlanes.bitpacked` format. Empty arrays use width zero. Differing widths serialize as
/// `fastlanes.bitpacked_v2`, with the byte-offset boundaries as a child.
#[derive(Debug, Clone)]
pub struct BitPackedPlugin;

impl ArrayPlugin for BitPackedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&BitPacked)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![self.id(), bitpacked_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            self.id() == array.encoding_id(),
            "array plugin {} cannot serialize in-memory array {}",
            self.id(),
            array.encoding_id(),
        );
        let view = array.as_::<BitPacked>();
        let widths = view.chunk_layout(&mut session.create_execution_ctx())?;
        if widths.is_uniform() {
            let metadata = BitPackedMetadata {
                bit_width: widths.max_width() as u32,
                offset: view.offset() as u32,
                patches: view
                    .patches()
                    .map(|p| p.to_metadata(view.len(), view.dtype()))
                    .transpose()?,
            }
            .encode_to_vec();
            let children = array.slots()[..BitPackedSlots::CHUNK_OFFSETS]
                .iter()
                .flatten()
                .cloned()
                .collect();
            return Ok(Some(ArraySerialization::new(
                self.id(),
                metadata,
                array.buffers(),
                children,
            )));
        }
        let metadata = BitPackedV2Metadata {
            offset: view.offset() as u32,
            patches: view
                .patches()
                .map(|p| p.to_metadata(view.len(), view.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        // The children run patches, validity, then chunk offsets.
        Ok(Some(ArraySerialization::from_array(
            bitpacked_v2_id(),
            array,
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if parts.serialized_id == self.id() {
            return deserialize_v1(parts);
        }
        vortex_ensure!(
            parts.serialized_id == bitpacked_v2_id(),
            "array plugin {} does not recognize serialized ID {}",
            self.id(),
            parts.serialized_id,
        );
        deserialize_v2(parts, session)
    }
}

/// Read the `fastlanes.bitpacked_v2` format: [`BitPackedV2Metadata`], one packed buffer, and
/// children running patches, validity, then chunk offsets.
fn deserialize_v2(
    parts: ArrayDeserialization<'_>,
    _session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let ArrayDeserialization {
        dtype,
        len,
        metadata,
        buffers,
        children,
        ..
    } = parts;
    let metadata = BitPackedV2Metadata::decode(metadata)?;
    let packed = single_buffer(buffers)?;
    let offset = offset_from_metadata(metadata.offset)?;
    let num_chunks = (len + offset as usize).div_ceil(FL_CHUNK_SIZE);
    let (patches, validity, table_idx) =
        deserialize_children(children, metadata.patches, dtype, len, 1)?;
    let offsets = children.get(table_idx, &CHUNK_OFFSETS_DTYPE, num_chunks + 1)?;
    Ok(BitPacked::try_new(
        packed,
        dtype.as_ptype(),
        validity,
        patches,
        offsets,
        len,
        offset,
    )?
    .into_array())
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayVTable;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::BitPackedMetadata;
    use super::BitPackedPlugin;
    use crate::BitPacked;
    use crate::BitPackedData;

    #[test]
    fn serialize_rejects_other_encodings() {
        let session = vortex_array::array_session();
        let values = PrimitiveArray::from_iter([1u32, 2, 3]).into_array();
        assert!(BitPackedPlugin.serialize(&values, &session).is_err());
    }

    #[rstest]
    #[case(2, u32::MAX, u32::MAX, "Expected 0 or 1 children")]
    #[case(0, u32::MAX, u32::MAX, "bit_width")]
    #[case(0, 0, u32::MAX, "offset")]
    fn invalid_inputs_preserve_validation_order(
        #[case] num_children: usize,
        #[case] bit_width: u32,
        #[case] offset: u32,
        #[case] expected: &str,
    ) -> VortexResult<()> {
        let session = vortex_array::array_session();
        let child = PrimitiveArray::from_iter([0u32]).into_array();
        let metadata = BitPackedMetadata {
            bit_width,
            offset,
            patches: None,
        }
        .encode_to_vec();
        let buffers = [BufferHandle::new_host(ByteBuffer::empty())];
        let children = vec![child.clone(); num_children];
        let error = BitPackedPlugin
            .deserialize(
                ArrayDeserialization::new(
                    BitPackedPlugin.id(),
                    child.dtype(),
                    0,
                    &metadata,
                    &buffers,
                    &children,
                ),
                &session,
            )
            .err()
            .ok_or_else(|| vortex_err!("Invalid inputs must be rejected"))?;
        assert!(error.to_string().contains(expected), "{error}");
        Ok(())
    }

    #[test]
    fn serde_requires_plugin() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(511), Some(7)]).into_array();
        let packed = BitPackedData::encode(&values, 3, &mut ctx)?;
        let array = packed.as_array();
        let serialized = BitPackedPlugin
            .serialize(array, &session)?
            .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
        let buffers = serialized
            .buffers
            .iter()
            .cloned()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        assert!(ArrayVTable::serialize(packed.as_view(), &session).is_err());
        assert!(
            ArrayVTable::deserialize(
                &BitPacked,
                array.dtype(),
                array.len(),
                &serialized.metadata,
                &buffers,
                &serialized.children,
                &session,
            )
            .is_err()
        );

        let decoded = BitPackedPlugin.deserialize(
            ArrayDeserialization::new(
                BitPackedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialized.metadata,
                &buffers,
                &serialized.children,
            ),
            &session,
        )?;
        assert_arrays_eq!(decoded, values, &mut ctx);
        Ok(())
    }
}
