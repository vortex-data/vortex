// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization plugin for [`Patched`].
//!
//! The chunk-local layout is written under the `vortex.patched_v2` wire ID. The retired
//! lane-transposed layout keeps `vortex.patched` for reading only: its patches are re-sorted into
//! chunk order on load, so no other code path ever sees lanes.

use prost::Message;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Patched;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::layout::n_chunks;
use crate::dtype::PType;
use crate::patches::PATCH_CHUNK_SIZE;
use crate::validity::Validity;

/// Wire ID of the chunk-local [`Patched`] layout.
pub fn patched_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.patched_v2");
    *ID
}

/// Serializes [`Patched`] arrays as `vortex.patched_v2` and deserializes both that and the
/// retired lane-transposed `vortex.patched` layout.
#[derive(Clone, Debug)]
pub struct PatchedPlugin;

impl ArrayPlugin for PatchedPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&Patched)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![ArrayVTable::id(&Patched), patched_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.encoding_id() == self.id(),
            "Patched plugin cannot serialize in-memory array {}",
            array.encoding_id(),
        );
        Ok(
            <Patched as ArrayVTable>::serialize(array.as_::<Patched>(), session)?
                .map(|metadata| ArraySerialization::from_array(patched_v2_id(), array, metadata)),
        )
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        if parts.serialized_id == patched_v2_id() {
            return Ok(Array::<Patched>::try_from_parts(ArrayVTable::deserialize(
                &Patched,
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
            parts.serialized_id == ArrayVTable::id(&Patched),
            "Patched plugin does not recognize serialized ID {}",
            parts.serialized_id,
        );
        deserialize_lanes(parts, session)
    }
}

/// Metadata of the retired lane-transposed layout.
#[derive(Clone, prost::Message)]
struct LanePatchedMetadata {
    #[prost(uint32, tag = "1")]
    n_patches: u32,
    #[prost(uint32, tag = "2")]
    n_lanes: u32,
    #[prost(uint32, tag = "3")]
    offset: u32,
}

/// Read a lane-transposed array and re-sort its patches into chunk order.
///
/// Lanes group each chunk's patches by `row % n_lanes`, so `lane_offsets[c * n_lanes]` is already
/// the prefix count for chunk `c`; only the order within each chunk has to be restored. Lane
/// offsets are absolute ordinals into the shared patch children and need not start at zero after
/// a chunk-granular slice.
fn deserialize_lanes(
    parts: ArrayDeserialization<'_>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let metadata = LanePatchedMetadata::decode(parts.metadata)?;
    let n_patches = metadata.n_patches as usize;
    let n_lanes = metadata.n_lanes as usize;
    let offset = metadata.offset as usize;
    vortex_ensure!(
        n_lanes > 0,
        "lane-transposed Patched array must have at least one lane"
    );
    vortex_ensure!(
        offset < PATCH_CHUNK_SIZE,
        "lane-transposed Patched offset {offset} must be within the first chunk"
    );
    let n_chunks = n_chunks(offset, parts.len);

    let inner = parts.children.get(0, parts.dtype, parts.len)?;
    let lane_offsets = parts
        .children
        .get(1, &PType::U32.into(), n_chunks * n_lanes + 1)?;
    let indices = parts.children.get(2, &PType::U16.into(), n_patches)?;
    let values = parts.children.get(3, parts.dtype, n_patches)?;

    let mut ctx = session.create_execution_ctx();
    let lane_offsets = lane_offsets.execute::<PrimitiveArray>(&mut ctx)?;
    let lane_offsets = lane_offsets.as_slice::<u32>();
    let indices = indices.execute::<PrimitiveArray>(&mut ctx)?;
    let indices = indices.as_slice::<u16>();

    let base = lane_offsets[0];
    let end = lane_offsets[n_chunks * n_lanes];
    vortex_ensure!(
        base <= end && end as usize <= n_patches,
        "lane offsets {base}..{end} exceed the {n_patches} patches"
    );
    let live = (end - base) as usize;
    let mut permutation: Vec<u32> = Vec::with_capacity(live);
    let mut sorted: Vec<u16> = Vec::with_capacity(live);
    let mut chunk_offsets: Vec<u32> = Vec::with_capacity(n_chunks + 1);
    for chunk in 0..n_chunks {
        let start = lane_offsets[chunk * n_lanes];
        let stop = lane_offsets[(chunk + 1) * n_lanes];
        vortex_ensure!(
            base <= start && start <= stop && stop <= end,
            "lane offsets of chunk {chunk} are not monotonic"
        );
        chunk_offsets.push(start - base);
        let mut ordinals: Vec<u32> = (start..stop).collect();
        ordinals.sort_unstable_by_key(|&ordinal| indices[ordinal as usize]);
        sorted.extend(ordinals.iter().map(|&ordinal| indices[ordinal as usize]));
        permutation.extend(ordinals);
    }
    chunk_offsets.push(end - base);

    let values = values
        .take(PrimitiveArray::new(Buffer::from(permutation), Validity::NonNullable).into_array())?;
    Ok(Patched::try_new(
        inner,
        PrimitiveArray::new(Buffer::from(sorted), Validity::NonNullable).into_array(),
        values,
        PrimitiveArray::new(Buffer::from(chunk_offsets), Validity::NonNullable).into_array(),
        offset,
    )?
    .into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::ArrayRef;
    use crate::ArrayVTable;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Bool;
    use crate::arrays::Patched;
    use crate::arrays::PatchedPlugin;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patched::PatchedArrayExt;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::arrays::patched::patched_v2_id;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::patches::Patches;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::session::ArraySessionExt;

    /// Arrays serialized by the lane-transposed `vortex.patched` encoding on `develop` before the
    /// chunk-local layout landed. The sources are rebuilt below with the current constructor so the
    /// two layouts can be compared; regenerate the bytes from a pre-change checkout if they ever
    /// need to change.
    const LANE_MULTI_CHUNK: &[u8] = include_bytes!("testdata/lane_multi_chunk.bin");
    const LANE_SLICED: &[u8] = include_bytes!("testdata/lane_sliced.bin");
    const LANE_NULLABLE: &[u8] = include_bytes!("testdata/lane_nullable.bin");

    fn decode_lanes(bytes: &[u8], dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let session = array_session();
        session.arrays().register(PatchedPlugin);
        let ids = ReadContext::new(vec![
            ArrayVTable::id(&Patched),
            ArrayVTable::id(&Primitive),
            ArrayVTable::id(&Bool),
        ]);
        SerializedArray::try_from(ByteBuffer::copy_from(bytes))?.decode(dtype, len, &ids, &session)
    }

    /// The `u16` source of the multi-chunk and sliced fixtures. Rows 1500 and 1531 sit in lanes
    /// 28 and 27, so the lane layout stored them in the opposite order from their rows.
    fn multi_chunk_source() -> VortexResult<ArrayRef> {
        let mut ctx = array_session().create_execution_ctx();
        let inner = PrimitiveArray::from_iter(0..2100u16).into_array();
        let patches = Patches::new(
            2100,
            0,
            buffer![5u32, 1030, 1031, 1500, 1531, 2099].into_array(),
            buffer![60001u16, 60002, 60003, 60004, 60005, 60006].into_array(),
            None,
        )?;
        Ok(Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array())
    }

    #[test]
    fn reads_lane_layout_into_chunk_order() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::U16, Nullability::NonNullable);
        let decoded = decode_lanes(LANE_MULTI_CHUNK, &dtype, 2100)?;
        let patched = decoded.as_::<Patched>();

        assert_eq!(patched.offset(), 0);
        assert_eq!(
            patched.patch_indices().as_::<Primitive>().as_slice::<u16>(),
            &[5, 6, 7, 476, 507, 51]
        );
        assert_eq!(
            patched.chunk_offsets().as_::<Primitive>().as_slice::<u32>(),
            &[0, 1, 5, 6]
        );

        assert_arrays_eq!(
            multi_chunk_source()?.execute::<PrimitiveArray>(&mut ctx)?,
            decoded.execute::<PrimitiveArray>(&mut ctx)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn reads_sliced_lane_layout() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::U16, Nullability::NonNullable);
        // The lane layout sliced its offsets at chunk granularity and shared the patch children,
        // so the fixture starts mid-chunk with a dead patch (row 1030) ahead of the first row.
        let decoded = decode_lanes(LANE_SLICED, &dtype, 2000 - 1031)?;
        let patched = decoded.as_::<Patched>();

        assert_eq!(patched.offset(), 7);
        assert_eq!(
            patched.patch_indices().as_::<Primitive>().as_slice::<u16>(),
            &[6, 7, 476, 507]
        );
        assert_eq!(
            patched.chunk_offsets().as_::<Primitive>().as_slice::<u32>(),
            &[0, 4]
        );

        assert_arrays_eq!(
            multi_chunk_source()?
                .slice(1031..2000)?
                .execute::<PrimitiveArray>(&mut ctx)?,
            decoded.execute::<PrimitiveArray>(&mut ctx)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn reads_nullable_lane_layout() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::U8, Nullability::Nullable);
        let decoded = decode_lanes(LANE_NULLABLE, &dtype, 1100)?;
        assert!(decoded.is::<Patched>());

        let inner = PrimitiveArray::from_option_iter(
            (0..1100u32).map(|i| (i % 7 != 0).then(|| u8::try_from(i % 256).vortex_expect("fits"))),
        )
        .into_array();
        let patches = Patches::new(
            1100,
            0,
            PrimitiveArray::from_iter([1u32, 1024, 1025, 1026]).into_array(),
            PrimitiveArray::from_option_iter([Some(201u8), Some(202), Some(203), Some(204)])
                .into_array(),
            None,
        )?;
        let expected = Patched::from_array_and_patches(inner, &patches, &mut ctx)?.into_array();

        assert_arrays_eq!(
            expected.execute::<PrimitiveArray>(&mut ctx)?,
            decoded.execute::<PrimitiveArray>(&mut ctx)?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn writes_the_v2_wire_id() -> VortexResult<()> {
        let array = multi_chunk_source()?;
        let session = array_session();
        session.arrays().register(PatchedPlugin);

        let ctx = ArrayContext::empty().with_allowed_ids(
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect()),
        );
        let serialized = array.serialize(&ctx, &session, &SerializeOptions::default())?;
        assert!(ctx.to_ids().contains(&patched_v2_id()));
        assert!(!ctx.to_ids().contains(&ArrayVTable::id(&Patched)));

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let decoded = SerializedArray::try_from(concat.freeze())?.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(ctx.to_ids()),
            &session,
        )?;
        assert!(decoded.is::<Patched>());
        Ok(())
    }
}
