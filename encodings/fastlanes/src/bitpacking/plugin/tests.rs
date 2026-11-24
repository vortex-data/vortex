// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PatchedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::BitPackedPatchedPlugin;
use super::BitPackedPlugin;
use super::v1::BitPackedMetadata;
use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

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
            ArrayVTable::id(&BitPacked),
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
            ArrayVTable::id(&BitPacked),
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
            ArrayVTable::id(&BitPacked),
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
    assert_eq!(serialization.serialized_id, ArrayVTable::id(&BitPacked));
    let metadata = BitPackedMetadata::decode(serialization.metadata.as_slice())?;
    assert_eq!(metadata.bit_width, u32::from(bit_width));
    assert_eq!(metadata.offset, u32::from(view.offset()));
    assert_eq!(metadata.patches.is_some(), view.patches().is_some());

    let read = roundtrip(&array)?;
    assert_eq!(read.encoding_id(), BitPackedPlugin.id());
    assert_arrays_eq!(read, values.slice(range)?, &mut ctx);
    Ok(())
}

/// Slicing removes individual patches while retaining the first chunk's original offset.
#[rstest]
fn serde_roundtrip_sliced_chunked_patches(#[values(500, 1500)] start: usize) -> VortexResult<()> {
    let mut ctx = PLUGIN_SESSION.create_execution_ctx();
    // Every 7th value does not fit 11 bits and becomes a patch, in every chunk.
    let values =
        PrimitiveArray::from_iter((0u32..4096).map(|i| if i % 7 == 0 { 5000 } else { i % 2048 }));
    let encoded = BitPackedData::encode(&values.clone().into_array(), 11, &mut ctx)?;
    // Slice mid-chunk while still encoded, as the slice kernel does.
    let sliced = <BitPacked as SliceKernel>::slice(encoded.as_view(), start..4096, &mut ctx)?
        .ok_or_else(|| vortex_err!("BitPacked slices"))?;
    let sliced = sliced.optimize_ctx(&PLUGIN_SESSION)?;
    let read = roundtrip(&sliced)?;
    assert_eq!(read.as_::<BitPacked>().bit_width(), 11);
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
