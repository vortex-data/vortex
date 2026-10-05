// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementations for `BitPacked`.
//!
//! [`BitPackedPlugin`] owns serde for `fastlanes.bitpacked`. [`BitPackedPatchedPlugin`] lets you
//! load in and deserialize a `BitPacked` array with interior patches as a `PatchedArray` that wraps
//! a patchless `BitPacked` array.
//!
//! This enables zero-cost backward compatibility with previously written datasets.

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
use vortex_array::arrays::Patched;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

/// Serde for the [`BitPacked`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde. Direct registration of
/// [`BitPacked`] does not support serde.
#[derive(Clone, Debug)]
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
        let view = array.as_opt::<BitPacked>().ok_or_else(|| {
            vortex_err!("BitPacked plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = BitPackedMetadata {
            bit_width: view.bit_width() as u32,
            offset: view.offset() as u32,
            patches: view
                .patches()
                .map(|p| p.to_metadata(view.len(), view.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            self.id(),
            array,
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            self.id(),
            "BitPacked plugin does not recognize serialized ID"
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

        let slots = {
            let mut s = ArraySlots::with_capacity(4);
            PatchesData::push_slots(&mut s, patches.as_ref());
            s.push(validity_to_child(&validity, len));
            s
        };
        let data = BitPackedData::try_new(
            packed,
            patches,
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
        Ok(Array::<BitPacked>::try_from_parts(
            ArrayParts::new(BitPacked, dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}

/// Custom deserialization plugin that converts a BitPacked array with interior
/// Patches into a PatchedArray holding a BitPacked array.
#[derive(Debug, Clone)]
pub(crate) struct BitPackedPatchedPlugin;

impl ArrayPlugin for BitPackedPatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `BitPacked` ID so that we can take over its
        // deserialization pathway.
        // TODO(joe): dedup method name
        ArrayVTable::id(&BitPacked)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        // delegate to BitPackedPlugin for serialization
        BitPackedPlugin.serialize(array, session)
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            self.id(),
            "BitPacked plugin does not recognize serialized ID"
        );
        let bitpacked: BitPackedArray = BitPackedPlugin
            .deserialize(parts, session)?
            .try_downcast()
            .map_err(|_| {
                vortex_err!("BitPacked plugin should only deserialize fastlanes.bitpacked")
            })?;

        // Create a new BitPackedArray without the interior patches installed.
        let Some(patches) = bitpacked.patches() else {
            return Ok(bitpacked.into_array());
        };

        let packed = bitpacked.packed().clone();
        let ptype = bitpacked.dtype().as_ptype();
        let validity = bitpacked.validity()?;
        let bw = bitpacked.bit_width;
        let len = bitpacked.len();
        let offset = bitpacked.offset();

        let bitpacked_without_patches =
            BitPacked::try_new(packed, ptype, validity, None, bw, len, offset)?.into_array();

        let patched = Patched::from_array_and_patches(
            bitpacked_without_patches,
            &patches,
            &mut session.create_execution_ctx(),
        )?;

        Ok(patched.into_array())
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        id == ArrayVTable::id(&BitPacked) || id == ArrayVTable::id(&Patched)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::BitPackedMetadata;
    use super::BitPackedPatchedPlugin;
    use super::BitPackedPlugin;
    use crate::BitPacked;
    use crate::BitPackedArray;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        session.arrays().register(BitPackedPatchedPlugin);
        session
    });

    #[test]
    fn test_decode_bitpacked_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Create values where some exceed the bit width, causing patches.
        // With bit_width=9, max value is 511. Values >=512 become patches.
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.into_array();
        let bitpacked = BitPackedData::encode(&parray, 9, &mut ctx)?;

        assert!(
            bitpacked.patches().is_some(),
            "Expected BitPacked array to have patches"
        );

        let array = bitpacked.as_array();

        let serialization = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = BitPackedPatchedPlugin.deserialize(
            ArrayDeserialization::new(
                BitPackedPatchedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        )?;

        let patched: PatchedArray = deserialized
            .try_downcast()
            .map_err(|a| vortex_err!("Expected Patched, got {}", a.encoding_id()))?;

        let inner_bitpacked: BitPackedArray = patched
            .inner()
            .clone()
            .try_downcast()
            .map_err(|a| vortex_err!("Expected inner BitPacked, got {}", a.encoding_id()))?;

        assert!(
            inner_bitpacked.patches().is_none(),
            "Inner BitPacked should NOT have patches"
        );

        Ok(())
    }

    #[test]
    fn bitpacked_without_patches_stays_bitpacked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // With bit_width=16, max value is 65535. All values 0..100 fit.
        let values: Buffer<i32> = (0i32..100).collect();
        let parray = values.into_array();
        let bitpacked = BitPackedData::encode(&parray, 16, &mut ctx)?;

        assert!(
            bitpacked.patches().is_none(),
            "Expected BitPacked array without patches"
        );

        let array = bitpacked.as_array();

        let serialization = SESSION.array_serialize(array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let deserialized = BitPackedPatchedPlugin.deserialize(
            ArrayDeserialization::new(
                BitPackedPatchedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        )?;

        let result = deserialized
            .try_downcast::<BitPacked>()
            .map_err(|a| vortex_err!("Expected deserialize BitPacked, got {}", a.encoding_id()))?;

        assert!(result.patches().is_none(), "Result should not have patches");

        Ok(())
    }

    #[test]
    fn primitive_array_returns_error() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

        let serialization = SESSION.array_serialize(&array)?.unwrap();
        let children = array.children();
        let buffers = array
            .buffers()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        let result = BitPackedPatchedPlugin.deserialize(
            ArrayDeserialization::new(
                BitPackedPatchedPlugin.id(),
                array.dtype(),
                array.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        );

        assert!(
            result.is_err(),
            "Expected error when deserializing PrimitiveArray with BitPackedPatchedPlugin"
        );

        Ok(())
    }

    static PLUGIN_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        session.arrays().register(BitPackedPlugin);
        session
    });

    fn roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
        let array_ctx = ArrayContext::empty();
        let buffers = array.serialize(&array_ctx, &PLUGIN_SESSION, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in buffers {
            bytes.extend_from_slice(&buffer);
        }
        SerializedArray::try_from(bytes.freeze())?.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(array_ctx.to_ids()),
            &PLUGIN_SESSION,
        )
    }

    #[rstest]
    #[case::no_patches(PrimitiveArray::from_iter(0u32..3000).into_array(), 12, 0..3000)]
    #[case::patches(PrimitiveArray::from_iter(0i32..=2048).into_array(), 9, 0..2049)]
    #[case::nullable(
        PrimitiveArray::from_option_iter((0u16..3000).map(|i| (i % 5 != 0).then_some(i))).into_array(),
        8,
        0..3000,
    )]
    #[case::sliced(PrimitiveArray::from_iter(0u32..3000).into_array(), 12, 700..1900)]
    fn serde_roundtrip(
        #[case] values: ArrayRef,
        #[case] bit_width: u8,
        #[case] range: std::ops::Range<usize>,
    ) -> VortexResult<()> {
        let mut ctx = PLUGIN_SESSION.create_execution_ctx();
        let array = BitPackedData::encode(&values, bit_width, &mut ctx)?
            .into_array()
            .slice(range.clone())?;
        let view = array.as_::<BitPacked>();
        let serialization = PLUGIN_SESSION
            .array_serialize(&array)?
            .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
        assert_eq!(serialization.serialized_id, BitPackedPlugin.id());
        let metadata = BitPackedMetadata::decode(serialization.metadata.as_slice())?;
        assert_eq!(metadata.bit_width, u32::from(bit_width));
        assert_eq!(metadata.offset, u32::from(view.offset()));
        assert_eq!(metadata.patches.is_some(), view.patches().is_some());

        let read = roundtrip(&array)?;
        assert_eq!(read.encoding_id(), BitPackedPlugin.id());
        assert_arrays_eq!(read, values.slice(range)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn vtable_serde_requires_plugin() -> VortexResult<()> {
        let values = PrimitiveArray::from_iter([1u8, 2, 3]).into_array();
        let array = BitPackedData::encode(&values, 2, &mut SESSION.create_execution_ctx())?;
        let session = vortex_array::array_session();
        session.arrays().register(BitPacked);
        assert!(session.array_serialize(&array.into_array()).is_err());
        Ok(())
    }
}
