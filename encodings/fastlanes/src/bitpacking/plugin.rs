// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization plugins for the frozen bit-packed format and its external-patches adapter.

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

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
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
    let offset = offset_from_metadata(metadata.offset)?;
    let bit_width = u8::try_from(metadata.bit_width).map_err(|_| {
        vortex_err!(
            "BitPackedMetadata bit_width {} does not fit in u8",
            metadata.bit_width
        )
    })?;
    let (patches, validity, _) = deserialize_children(children, metadata.patches, dtype, len, 0)?;

    let slots = {
        let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
        PatchesData::push_slots(&mut s, patches.as_ref());
        s.push(validity_to_child(&validity, len));
        s
    };
    let data = BitPackedData::try_new(packed, patches, bit_width, offset)?;
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
        vortex_ensure!(
            parts.serialized_id == self.id(),
            "BitPacked plugin does not recognize serialized ID {}",
            parts.serialized_id,
        );
        deserialize_v1(parts)
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
        // Both plugins share the same wire contract.
        BitPackedPlugin.serialize(array, session)
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let bitpacked = BitPackedPlugin.deserialize(parts, session)?;
        let bitpacked = bitpacked.as_::<BitPacked>().into_owned();

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

    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayVTable;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

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
    #[test]
    fn serde_requires_plugin() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(511), Some(7)]).into_array();
        let packed = BitPackedData::encode(&values, 3, &mut ctx)?;
        let array = packed.as_array();
        let serialized = BitPackedPlugin
            .serialize(array, &SESSION)?
            .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
        let buffers = serialized
            .buffers
            .iter()
            .cloned()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();

        assert!(ArrayVTable::serialize(packed.as_view(), &SESSION).is_err());
        assert!(
            ArrayVTable::deserialize(
                &BitPacked,
                array.dtype(),
                array.len(),
                &serialized.metadata,
                &buffers,
                &serialized.children,
                &SESSION,
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
            &SESSION,
        )?;
        assert_arrays_eq!(decoded, values, &mut ctx);
        Ok(())
    }
}
