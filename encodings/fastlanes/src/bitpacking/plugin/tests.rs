// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PatchedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::BitPackedPatchedPlugin;
use super::BitPackedPlugin;
use super::bitpacked_v1_id;
use super::bitpacked_v2_id;
use super::v1::BitPackedMetadata;
use super::v2::BitPackedV2Metadata;
use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::BitPackedData;
use crate::BitWidthsView;
use crate::bitpack_compress::bitpack_blocked_to_best_bit_widths;
use crate::bitpack_compress::bitpack_encode_blocked;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
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
            bitpacked_v1_id(),
            array.dtype(),
            array.len(),
            &serialization.metadata,
            &buffers,
            &children,
        ),
        &SESSION,
    )?;

    assert_arrays_eq!(deserialized, parray, &mut ctx);

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
            bitpacked_v1_id(),
            array.dtype(),
            array.len(),
            &serialization.metadata,
            &buffers,
            &children,
        ),
        &SESSION,
    )?;

    assert_arrays_eq!(deserialized, parray, &mut ctx);

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
            bitpacked_v1_id(),
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
    let session = VortexSession::empty();
    crate::initialize(&session);
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
        .slice(range.clone())?
        .optimize_ctx(&PLUGIN_SESSION)?;
    let view = array.as_::<BitPacked>();
    let serialization = PLUGIN_SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    assert_eq!(serialization.serialized_id, bitpacked_v1_id());
    let metadata = BitPackedMetadata::decode(serialization.metadata.as_slice())?;
    assert_eq!(metadata.bit_width, u32::from(bit_width));
    assert_eq!(metadata.offset, u32::from(view.offset()));
    assert_eq!(metadata.patches.is_some(), view.patches().is_some());

    let read = roundtrip(&array)?;
    assert_eq!(read.encoding_id(), BitPackedPlugin.id());
    assert_arrays_eq!(read, values.slice(range)?, &mut ctx);
    Ok(())
}

/// Values whose 1024-value blocks need 2 to 6 bits.
fn drifting() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..5000u32).map(|i| i % (4 << (i / 1024))))
}

/// [`drifting`] with every 7th value null.
fn nullable_drifting() -> PrimitiveArray {
    PrimitiveArray::from_option_iter(
        (0..5000u32).map(|i| (i % 7 != 0).then_some(i % (4 << (i / 1024)))),
    )
}

/// Host buffer handles for the buffers of `serialization`.
fn buffer_handles(serialization: &ArraySerialization) -> Vec<BufferHandle> {
    serialization
        .buffers
        .iter()
        .cloned()
        .map(BufferHandle::new_host)
        .collect()
}

#[rstest]
#[case::no_patches(drifting(), None)]
#[case::patches(drifting(), Some(vec![2, 3, 4, 5, 3]))]
#[case::nullable(nullable_drifting(), None)]
#[case::nullable_patches(nullable_drifting(), Some(vec![2, 3, 4, 5, 3]))]
fn v2_roundtrip(
    #[case] values: PrimitiveArray,
    #[case] bit_widths: Option<Vec<u8>>,
) -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    let array = match bit_widths {
        Some(bit_widths) => bitpack_encode_blocked(&values, &bit_widths, None, &mut ctx)?,
        None => bitpack_blocked_to_best_bit_widths(&values, &mut ctx)?,
    };
    let BitWidthsView::Blocked(block_offsets) = array.bit_widths() else {
        vortex_bail!("expected block offsets");
    };
    let serialization = PLUGIN_SESSION
        .array_serialize(array.as_array())?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    assert_eq!(serialization.serialized_id, bitpacked_v2_id());
    let metadata = BitPackedV2Metadata::decode(serialization.metadata.as_slice())?;
    assert_eq!(metadata.offset, 0);
    assert_eq!(
        metadata.block_offsets_ptype,
        PType::try_from(block_offsets.dtype())? as i32
    );
    assert_eq!(metadata.patches.is_some(), array.patches().is_some());
    assert_eq!(metadata.has_validity, array.validity_child().is_some());
    // The block offsets come first, then the patches, then the validity.
    let patch_children = array
        .patches()
        .map_or(0, |p| 2 + usize::from(p.chunk_offsets().is_some()));
    assert_eq!(
        serialization.children.len(),
        1 + patch_children + usize::from(metadata.has_validity)
    );
    assert!(ArrayRef::ptr_eq(&serialization.children[0], block_offsets));
    if let Some(patches) = array.patches() {
        assert!(ArrayRef::ptr_eq(
            &serialization.children[1],
            patches.indices()
        ));
        assert!(ArrayRef::ptr_eq(
            &serialization.children[2],
            patches.values()
        ));
    }
    if let Some(validity) = array.validity_child() {
        assert!(ArrayRef::ptr_eq(
            &serialization.children[1 + patch_children],
            validity
        ));
    }

    let read = roundtrip(array.as_array())?;
    assert_eq!(read.encoding_id(), bitpacked_v2_id());
    assert!(!read.as_::<BitPacked>().bit_widths().is_global());
    assert_arrays_eq!(read, values, &mut ctx);
    Ok(())
}

#[test]
fn v2_roundtrip_offset_and_nonzero_base() -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    let values = drifting();
    let encoded = bitpack_blocked_to_best_bit_widths(&values, &mut ctx)?;
    let BitWidthsView::Blocked(block_offsets) = encoded.bit_widths() else {
        vortex_bail!("expected block offsets");
    };
    let block_offsets = block_offsets
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    let block_offsets =
        PrimitiveArray::from_iter(block_offsets.as_slice::<u64>().iter().map(|o| o + 128));
    let array = BitPacked::try_new_with_block_offsets(
        encoded.packed().clone(),
        PType::U32,
        Validity::NonNullable,
        None,
        block_offsets.into_array(),
        4900,
        17,
    )?
    .into_array();

    let serialization = PLUGIN_SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    assert_eq!(
        BitPackedV2Metadata::decode(serialization.metadata.as_slice())?.offset,
        17
    );
    assert_arrays_eq!(
        roundtrip(&array)?,
        values.into_array().slice(17..4917)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn v2_roundtrip_patches_without_chunk_offsets() -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    let values = nullable_drifting();
    let encoded = bitpack_encode_blocked(&values, &[2, 3, 4, 5, 3], None, &mut ctx)?;
    let BitWidthsView::Blocked(block_offsets) = encoded.bit_widths() else {
        vortex_bail!("expected block offsets");
    };
    let patches = encoded
        .patches()
        .ok_or_else(|| vortex_err!("expected patches"))?;
    let patches = Patches::new(
        encoded.len(),
        0,
        patches.indices().clone(),
        patches.values().clone(),
        None,
        &mut ctx,
    )?;
    let array = BitPacked::try_new_with_block_offsets(
        encoded.packed().clone(),
        PType::U32,
        encoded.validity()?,
        Some(patches),
        block_offsets.clone(),
        encoded.len(),
        0,
    )?
    .into_array();

    let serialization = PLUGIN_SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    let metadata = BitPackedV2Metadata::decode(serialization.metadata.as_slice())?;
    assert!(
        metadata
            .patches
            .ok_or_else(|| vortex_err!("expected patches metadata"))?
            .chunk_offsets_dtype()?
            .is_none()
    );
    assert!(metadata.has_validity);
    // Block offsets, patch indices, patch values and validity.
    assert_eq!(serialization.children.len(), 4);

    assert_arrays_eq!(roundtrip(&array)?, values, &mut ctx);
    Ok(())
}

#[test]
fn v2_ignores_trailing_children() -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    let values = nullable_drifting();
    let array = bitpack_encode_blocked(&values, &[2, 3, 4, 5, 3], None, &mut ctx)?.into_array();
    let serialization = PLUGIN_SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    let buffers = buffer_handles(&serialization);
    let mut children = serialization.children;
    children.push(PrimitiveArray::from_iter([7u8, 8, 9]).into_array());

    let read = BitPackedPlugin.deserialize(
        ArrayDeserialization::new(
            bitpacked_v2_id(),
            array.dtype(),
            array.len(),
            &serialization.metadata,
            &buffers,
            &children,
        ),
        &PLUGIN_SESSION,
    )?;
    assert_arrays_eq!(read, values, &mut ctx);
    Ok(())
}

#[test]
fn v2_rejects_malformed_parts() -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    // Nullable with chunked patches, so every optional child is present.
    let array = bitpack_encode_blocked(&nullable_drifting(), &[2, 3, 4, 5, 3], None, &mut ctx)?
        .into_array();
    let serialization = PLUGIN_SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    let metadata = BitPackedV2Metadata::decode(serialization.metadata.as_slice())?;
    let patches = metadata
        .patches
        .ok_or_else(|| vortex_err!("expected patches metadata"))?;
    let buffers = buffer_handles(&serialization);
    let children = serialization.children;
    assert_eq!(children.len(), 5);
    let dtype = array.dtype().clone();
    let len = array.len();
    let deserialize = |dtype: &DType,
                       len: usize,
                       metadata: &BitPackedV2Metadata,
                       buffers: &[BufferHandle],
                       children: &[ArrayRef]| {
        BitPackedPlugin.deserialize(
            ArrayDeserialization::new(
                bitpacked_v2_id(),
                dtype,
                len,
                &metadata.encode_to_vec(),
                buffers,
                &children,
            ),
            &PLUGIN_SESSION,
        )
    };
    let with_patches = |patches: PatchesMetadata| BitPackedV2Metadata {
        patches: Some(patches),
        ..metadata.clone()
    };

    assert!(deserialize(&dtype, len, &metadata, &buffers, &children).is_ok());

    // Each child the metadata declares must be present.
    assert!(deserialize(&dtype, len, &metadata, &buffers, &children[..4]).is_err());
    assert!(deserialize(&dtype, len, &metadata, &buffers, &[]).is_err());
    assert!(deserialize(&dtype, len, &metadata, &[], &children).is_err());

    // These guards run before any child is read, so they must not reach the length and dtype
    // assertions of in-memory children.
    let utf8 = DType::Utf8(Nullability::Nullable);
    assert!(deserialize(&utf8, len, &metadata, &buffers, &children).is_err());
    let float = DType::Primitive(PType::F32, Nullability::Nullable);
    assert!(deserialize(&float, len, &metadata, &buffers, &children).is_err());
    assert!(deserialize(&dtype.as_nonnullable(), len, &metadata, &buffers, &children).is_err());
    let offset_past_block = BitPackedV2Metadata {
        offset: 1024,
        ..metadata.clone()
    };
    assert!(deserialize(&dtype, len, &offset_past_block, &buffers, &children).is_err());
    let signed_block_offsets = BitPackedV2Metadata {
        block_offsets_ptype: PType::I32 as i32,
        ..metadata.clone()
    };
    assert!(deserialize(&dtype, len, &signed_block_offsets, &buffers, &children).is_err());
    let unknown_block_offsets = BitPackedV2Metadata {
        block_offsets_ptype: 11,
        ..metadata.clone()
    };
    assert!(deserialize(&dtype, len, &unknown_block_offsets, &buffers, &children).is_err());
    assert!(deserialize(&dtype, usize::MAX, &metadata, &buffers, &children).is_err());

    // The packed buffer must fit the blocks at the widest bit width, in whole 128-byte rows.
    let max_packed_len = len.div_ceil(1024) * 128 * 32;
    for packed_len in [100, max_packed_len + 128] {
        let packed = [BufferHandle::new_host(ByteBuffer::from(vec![
            0u8;
            packed_len
        ]))];
        assert!(deserialize(&dtype, len, &metadata, &packed, &children).is_err());
    }

    // Patch counts must be in `1..=len`, and chunk offsets unsigned with at most one more chunk
    // than the array spans.
    let chunk_offsets_ptype = Some(PType::U64);
    let chunk_offsets_len = Some(len.div_ceil(1024));
    for bad_patches in [
        PatchesMetadata::new(
            0,
            0,
            PType::U16,
            chunk_offsets_len,
            chunk_offsets_ptype,
            Some(0),
        ),
        PatchesMetadata::new(
            len + 1,
            0,
            PType::U16,
            chunk_offsets_len,
            chunk_offsets_ptype,
            Some(0),
        ),
        PatchesMetadata::new(
            patches.len()?,
            0,
            PType::I16,
            chunk_offsets_len,
            chunk_offsets_ptype,
            Some(0),
        ),
        PatchesMetadata::new(
            patches.len()?,
            0,
            PType::U16,
            chunk_offsets_len,
            Some(PType::I64),
            Some(0),
        ),
        PatchesMetadata::new(
            patches.len()?,
            0,
            PType::U16,
            Some(0),
            chunk_offsets_ptype,
            Some(0),
        ),
        PatchesMetadata::new(
            patches.len()?,
            0,
            PType::U16,
            Some(len.div_ceil(1024) + 2),
            chunk_offsets_ptype,
            Some(0),
        ),
    ] {
        assert!(deserialize(&dtype, len, &with_patches(bad_patches), &buffers, &children).is_err());
    }

    // Host block offsets, the first child, must span the packed buffer.
    let mut boundaries = children[0]
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(&mut ctx)?
        .as_slice::<u64>()
        .to_vec();
    *boundaries
        .last_mut()
        .ok_or_else(|| vortex_err!("expected boundaries"))? += 128;
    let mut corrupt_children = children;
    corrupt_children[0] = PrimitiveArray::from_iter(boundaries).into_array();
    let u64_block_offsets = BitPackedV2Metadata {
        block_offsets_ptype: PType::U64 as i32,
        ..metadata.clone()
    };
    assert!(deserialize(&dtype, len, &u64_block_offsets, &buffers, &corrupt_children).is_err());
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_bitpacked_v2_metadata() {
    check_metadata(
        "bitpacked_v2.metadata",
        &BitPackedV2Metadata {
            offset: 1023,
            block_offsets_ptype: PType::U64 as i32,
            patches: Some(PatchesMetadata::new(
                3,
                5,
                PType::U16,
                Some(2),
                Some(PType::U32),
                Some(1),
            )),
            has_validity: true,
        }
        .encode_to_vec(),
    );
}

#[test]
fn test_decode_blocked_bitpacked_patches() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = drifting();
    let array = bitpack_encode_blocked(&values, &[2, 3, 4, 5, 3], None, &mut ctx)?.into_array();

    let serialization = SESSION.array_serialize(&array)?.unwrap();
    assert_eq!(serialization.serialized_id, bitpacked_v2_id());
    let buffers = buffer_handles(&serialization);

    let deserialized = BitPackedPatchedPlugin.deserialize(
        ArrayDeserialization::new(
            bitpacked_v2_id(),
            array.dtype(),
            array.len(),
            &serialization.metadata,
            &buffers,
            &serialization.children,
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
    assert!(inner_bitpacked.patches().is_none());
    assert!(!inner_bitpacked.bit_widths().is_global());
    assert_arrays_eq!(patched, values, &mut ctx);
    Ok(())
}

/// Slicing removes individual patches while retaining the first chunk's original offset.
#[rstest]
#[case::v1(false)]
#[case::v2(true)]
fn serde_roundtrip_sliced_chunked_patches(
    #[case] blocked: bool,
    #[values(500, 1500)] start: usize,
) -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    // Every 7th value does not fit 11 bits and becomes a patch, in every chunk.
    let values =
        PrimitiveArray::from_iter((0u32..4096).map(|i| if i % 7 == 0 { 5000 } else { i % 2048 }));
    let encoded = BitPackedData::encode(&values.clone().into_array(), 11, &mut ctx)?;
    // Slice mid-chunk while still encoded, as the slice kernel does.
    let sliced = <BitPacked as SliceKernel>::slice(encoded.as_view(), start..4096, &mut ctx)?
        .ok_or_else(|| vortex_err!("BitPacked slices"))?;
    let sliced = if blocked {
        // Represent the same sliced blocks as v2 without relying on a blocked slice kernel.
        let view = sliced.as_::<BitPacked>();
        let num_blocks = (usize::from(view.offset()) + view.len()).div_ceil(1024);
        let block_offsets = PrimitiveArray::from_iter(
            (0..=u32::try_from(num_blocks)?).map(|block| block * 11 * 128),
        );
        BitPacked::try_new_with_block_offsets(
            view.packed().clone(),
            PType::U32,
            view.validity()?,
            view.patches(),
            block_offsets.into_array(),
            view.len(),
            view.offset(),
        )?
        .into_array()
    } else {
        sliced
    };
    let sliced = sliced.optimize_ctx(&PLUGIN_SESSION)?;
    let read = roundtrip(&sliced)?;
    assert_eq!(read.as_::<BitPacked>().bit_widths().is_global(), !blocked);
    let expected = values.into_array().slice(start..4096)?;
    // Probe the first patch of the second retained chunk (row 529 when start is 500).
    let patch_index = ((start / 1024 + 1) * 1024).next_multiple_of(7) - start;
    assert_eq!(
        read.execute_scalar(patch_index, &mut ctx)?,
        expected.execute_scalar(patch_index, &mut ctx)?
    );
    assert_arrays_eq!(read, expected, &mut ctx);

    let indices = PrimitiveArray::from_iter([
        u32::try_from(patch_index - 1)?,
        u32::try_from(patch_index)?,
        u32::try_from(patch_index + 1)?,
    ])
    .into_array();
    let taken = read
        .take(indices.clone())?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_arrays_eq!(taken, expected.take(indices)?, &mut ctx);

    let read_patches = read.as_::<BitPacked>().patches().unwrap();
    let expected_patches = sliced.as_::<BitPacked>().patches().unwrap();
    assert_eq!(
        read_patches.offset_within_chunk(),
        expected_patches.offset_within_chunk()
    );
    let resliced = read_patches.slice(800..read.len(), &mut ctx)?.unwrap();
    let expected_resliced = expected_patches
        .slice(800..sliced.len(), &mut ctx)?
        .unwrap();
    assert_eq!(
        resliced.offset_within_chunk(),
        expected_resliced.offset_within_chunk()
    );
    assert_arrays_eq!(resliced.indices(), expected_resliced.indices(), &mut ctx);
    assert_arrays_eq!(resliced.values(), expected_resliced.values(), &mut ctx);
    Ok(())
}
