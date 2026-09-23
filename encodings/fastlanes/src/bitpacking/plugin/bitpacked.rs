// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization plugin for the frozen bit-packed wire format.

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
use vortex_array::arrays::ConstantArray;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedSlots;

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

/// Serialization boundary for the frozen `fastlanes.bitpacked` wire format.
#[derive(Debug, Clone)]
pub struct BitPackedPlugin;

impl ArrayPlugin for BitPackedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&BitPacked)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            self.id() == array.encoding_id(),
            "array plugin {} cannot serialize in-memory array {}",
            self.id(),
            array.encoding_id(),
        );
        let view = array.as_::<BitPacked>();
        let metadata = BitPackedMetadata {
            bit_width: view.bit_width() as u32,
            offset: view.offset() as u32,
            patches: view
                .patches()
                .map(|p| p.to_metadata(view.len(), view.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        let children = array.slots()[..BitPackedSlots::WIDTH_TABLE]
            .iter()
            .flatten()
            .cloned()
            .collect();
        Ok(Some(ArraySerialization::new(
            self.id(),
            metadata,
            array.buffers(),
            children,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            self.id() == parts.serialized_id,
            "array plugin {} does not recognize serialized ID {}",
            self.id(),
            parts.serialized_id,
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;

        let metadata = BitPackedMetadata::decode(metadata)?;
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let packed = buffers[0].clone();

        let load_validity = |child_idx: usize| {
            if children.len() == child_idx {
                Ok(Validity::from(dtype.nullability()))
            } else if children.len() == child_idx + 1 {
                let validity = children.get(child_idx, &Validity::DTYPE, len)?;
                Ok(Validity::Array(validity))
            } else {
                vortex_bail!(
                    "Expected {} or {} children, got {}",
                    child_idx,
                    child_idx + 1,
                    children.len()
                );
            }
        };

        let validity_idx = match &metadata.patches {
            None => 0,
            Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
            Some(_) => 2,
        };

        let validity = load_validity(validity_idx)?;

        let patches = metadata
            .patches
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

        let data = BitPackedData::try_new(
            packed,
            patches.clone(),
            u8::try_from(metadata.bit_width).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata bit_width {} does not fit in u8",
                    metadata.bit_width
                )
            })?,
            u16::try_from(metadata.offset).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata offset {} does not fit in u16",
                    metadata.offset
                )
            })?,
        )?;
        let slots = {
            let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            let num_chunks = (len + data.offset() as usize).div_ceil(FL_CHUNK_SIZE);
            s.push(Some(
                ConstantArray::new(data.bit_width(), num_chunks).into_array(),
            ));
            s
        };
        Ok(Array::<BitPacked>::try_from_parts(
            ArrayParts::new(BitPacked, dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
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
