// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`]s for bit-packed arrays.
//!
//! [`BitPackedPlugin`] owns the wire history of `BitPacked`: the frozen `fastlanes.bitpacked`
//! format for arrays whose chunks share one width, and `fastlanes.bitpacked_v2`, whose width
//! table child carries one width per chunk. [`BitPackedPatchedPlugin`] reads both and lifts
//! interior patches into a `Patched` array.

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::PatchesMetadata;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::ChunkWidths;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::WIDTH_TABLE_DTYPE;
use crate::bitpacking::vtable::deserialize_children;
use crate::bitpacking::vtable::offset_from_metadata;
use crate::bitpacking::vtable::single_buffer;

/// The serialized format for arrays whose chunks do not all share one bit width.
///
/// The original `fastlanes.bitpacked` format carries a single `bit_width`, and readers of that
/// format assume every chunk uses it. Arrays with differing chunk widths therefore serialize under
/// this successor ID, which older readers reject as unknown instead of misreading.
pub fn bitpacked_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("fastlanes.bitpacked_v2");
    *ID
}

/// Metadata of the `fastlanes.bitpacked_v2` format. The chunk widths travel in the width table
/// child, so there is no width here.
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
/// Arrays whose chunks share one width serialize through the encoding's own serializer as the
/// frozen `fastlanes.bitpacked` format, byte for byte. Differing widths serialize as
/// `fastlanes.bitpacked_v2`, with the width table as the last child.
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
        let view = array.as_::<BitPacked>();
        if view.chunk_widths().is_uniform() {
            return ArrayPlugin::serialize(&BitPacked, array, session);
        }
        let metadata = BitPackedV2Metadata {
            offset: view.offset() as u32,
            patches: view
                .patches()
                .map(|p| p.to_metadata(view.len(), view.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        // The array's children already run patches, validity, then the width table.
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
            return Ok(Array::<BitPacked>::try_from_parts(ArrayVTable::deserialize(
                &BitPacked,
                parts.dtype,
                parts.len,
                parts.metadata,
                parts.buffers,
                parts.children,
                session,
            )?)?
            .into_array());
        }
        vortex_ensure!(
            parts.serialized_id == bitpacked_v2_id(),
            "BitPacked plugin does not recognize serialized ID {}",
            parts.serialized_id,
        );
        deserialize_v2(parts, session)
    }
}

/// Read the `fastlanes.bitpacked_v2` format: [`BitPackedV2Metadata`], one packed buffer, and
/// children running patches, validity, then the width table.
fn deserialize_v2(
    parts: ArrayDeserialization<'_>,
    session: &VortexSession,
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
    let table = children.get(table_idx, &WIDTH_TABLE_DTYPE, num_chunks)?;
    let widths = ChunkWidths::new(
        table
            .clone()
            .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?
            .into_buffer::<u8>(),
    );
    let array = BitPacked::try_new(
        packed,
        dtype.as_ptype(),
        validity,
        patches,
        widths,
        len,
        offset,
    )?;
    // A table whose entries all agree describes a uniform array, which carries no table in
    // memory. Otherwise keep the table as read, since a compressor may have re-encoded it.
    if array.chunk_widths().is_uniform() {
        return Ok(array.into_array());
    }
    Ok(BitPacked::with_width_table(array, table)?.into_array())
}

/// Custom deserialization plugin that converts a BitPacked array with interior
/// Patches into a PatchedArray holding a BitPacked array.
#[derive(Debug, Clone)]
pub(crate) struct BitPackedPatchedPlugin;

impl ArrayPlugin for BitPackedPatchedPlugin {
    fn id(&self) -> ArrayId {
        // We reuse the existing `BitPacked` ID so that we can take over its
        // deserialization pathway.
        BitPackedPlugin.id()
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        BitPackedPlugin.serialized_ids()
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
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
        let widths = bitpacked.chunk_widths().clone();
        let len = bitpacked.len();
        let offset = bitpacked.offset();

        let bitpacked_without_patches =
            BitPacked::try_new(packed, ptype, validity, None, widths, len, offset)?.into_array();

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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PatchedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::patched::PatchedArraySlotsExt;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::BitPackedPatchedPlugin;
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
}
