// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Behavioural tests for bit-packed arrays whose chunks are packed at different widths.

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
use vortex_array::compute::conformance::cast::test_cast_conformance;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::BitPackedPlugin;
use crate::ChunkWidths;
use crate::FL_CHUNK_SIZE;
use crate::bitpacked_v2_id;
use crate::bitpacking::bitpack_compress::bitpack_encode_with_widths;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;
use crate::bitpacking::plugin::BitPackedMetadata;
use crate::bitpacking::plugin::BitPackedV2Metadata;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Four full chunks plus a partial trailer, each chunk with a distinctly different magnitude:
/// 3-bit values, 12-bit values with a few 20-bit outliers, all zeros, 20-bit values, and a
/// 5-bit tail.
fn varied(len_tail: usize) -> Vec<u32> {
    (0..4 * FL_CHUNK_SIZE + len_tail)
        .map(|i| {
            let chunk = i / FL_CHUNK_SIZE;
            let pos = (i % FL_CHUNK_SIZE) as u32;
            match chunk {
                0 => pos % 8,
                1 if pos % 300 == 7 => 1 << 20 | pos,
                1 => pos % 4096,
                2 => 0,
                3 => (pos * 977) % (1 << 20),
                _ => pos % 32,
            }
        })
        .collect()
}

fn encode(values: &[u32]) -> VortexResult<BitPackedArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let widths = ChunkWidths::new(Buffer::from_iter(values.chunks(FL_CHUNK_SIZE).map(
        |chunk| {
            chunk
                .iter()
                .map(|v| (u32::BITS - v.leading_zeros()) as u8)
                .max()
                .unwrap_or(0)
        },
    )));
    bitpack_encode_with_widths(
        &PrimitiveArray::from_iter(values.iter().copied()),
        widths,
        &mut ctx,
    )
}

fn primitive(values: &[u32]) -> ArrayRef {
    PrimitiveArray::from_iter(values.iter().copied()).into_array()
}

#[test]
fn explicit_widths_including_full_width_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<u16> = (0..2 * FL_CHUNK_SIZE + 10)
        .map(|i| {
            if i < FL_CHUNK_SIZE {
                (i % 16) as u16
            } else {
                u16::MAX - i as u16
            }
        })
        .collect();
    let array = PrimitiveArray::from_iter(values.iter().copied());
    // Chunk 1 and the tail use the full 16 bits, which a single global width could never pick.
    let widths = ChunkWidths::new(buffer![4u8, 16, 16]);
    let packed = bitpack_encode_with_widths(&array, widths, &mut ctx)?;
    assert!(packed.patches().is_none());
    assert_eq!(
        packed
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .max_width(),
        16
    );
    assert_arrays_eq!(packed, array, &mut ctx);
    Ok(())
}

/// Serialize `array` through the session and read it back through the plugin registered for the
/// serialized ID, as a file reader would.
fn serde_roundtrip(array: &BitPackedArray) -> VortexResult<(ArrayId, Vec<u8>, ArrayRef)> {
    let array_ref = array.as_array();
    let serialization = SESSION
        .array_serialize(array_ref)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    let array_ctx = ArrayContext::empty();
    let buffers = array_ref.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(&buffer);
    }
    let read = SerializedArray::try_from(bytes.freeze())?.decode(
        array_ref.dtype(),
        array_ref.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;
    Ok((serialization.serialized_id, serialization.metadata, read))
}

/// Differing chunk widths serialize under the v2 ID with the width table as a child, and read
/// back with the same widths.
#[test]
fn differing_widths_serialize_as_v2() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?;
    let table = packed.width_table();
    assert_eq!(
        table
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .as_slice::<u8>(),
        packed
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .as_buffer()
            .as_slice()
    );
    let (id, metadata, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, bitpacked_v2_id());
    assert_eq!(BitPackedV2Metadata::decode(metadata.as_slice())?.offset, 0);
    assert_eq!(
        read.as_::<BitPacked>()
            .chunk_widths(&mut SESSION.create_execution_ctx())?,
        packed.chunk_widths(&mut SESSION.create_execution_ctx())?
    );
    assert_eq!(
        read.as_::<BitPacked>().width_table().len(),
        packed.width_table().len()
    );
    assert_arrays_eq!(read, primitive(&values), &mut ctx);
    Ok(())
}

/// One shared width serializes under the original ID with the original metadata and no width
/// table, byte for byte.
#[test]
fn uniform_widths_serialize_as_original_format() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<u32> = (0..3000).map(|i| i % 128).collect();
    let packed = encode(&values)?;
    assert!(packed.width_table().is::<Constant>());
    assert_eq!(packed.as_array().children().len(), 2);
    assert!(
        SESSION
            .array_serialize(packed.as_array())?
            .ok_or_else(|| vortex_err!("must serialize"))?
            .children
            .is_empty()
    );
    let (id, metadata, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, ArrayVTable::id(&BitPacked));
    let original = BitPackedMetadata {
        bit_width: 7,
        offset: 0,
        patches: None,
    }
    .encode_to_vec();
    assert_eq!(metadata, original);
    assert_arrays_eq!(read, primitive(&values), &mut ctx);
    Ok(())
}

/// An array with no chunks has nothing to tabulate and stays in the original format.
#[test]
fn empty_array_serializes_as_original_format() -> VortexResult<()> {
    let packed = encode(&[])?;
    assert!(packed.width_table().is::<Constant>());
    let (id, _, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, ArrayVTable::id(&BitPacked));
    assert!(read.is_empty());
    Ok(())
}

/// A compressor may re-encode the width table. The re-encoded child survives a round trip and
/// still yields the same widths.
#[test]
fn compressed_width_table_round_trips() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?;
    let table = PrimitiveArray::new(
        packed
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .as_buffer(),
        Validity::NonNullable,
    );
    let compressed_table = bitpack_to_best_bit_width(&table, &mut ctx)?.into_array();
    assert!(compressed_table.is::<BitPacked>());
    let packed = BitPacked::with_width_table(packed, compressed_table)?;
    let (id, _, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, bitpacked_v2_id());
    let view = read.as_::<BitPacked>();
    assert_eq!(
        view.chunk_widths(&mut SESSION.create_execution_ctx())?,
        packed.chunk_widths(&mut SESSION.create_execution_ctx())?
    );
    assert!(view.width_table().is::<BitPacked>());
    assert_arrays_eq!(read, primitive(&values), &mut ctx);
    Ok(())
}

#[test]
fn widths_must_agree_with_offsets() -> VortexResult<()> {
    let packed = encode(&varied(0))?;
    let table = PrimitiveArray::from_iter(
        packed
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .as_buffer()
            .as_slice()
            .iter()
            .rev()
            .copied(),
    )
    .into_array();
    assert!(BitPacked::with_width_table(packed, table).is_err());
    Ok(())
}

#[rstest]
#[case::too_wide(64u8)]
#[case::wrong_packed_size(1u8)]
fn compressed_widths_are_validated_on_execution(#[case] width: u8) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let packed = encode(&varied(0))?;
    let mut serialized = SESSION
        .array_serialize(packed.as_array())?
        .ok_or_else(|| vortex_err!("must serialize"))?;
    let table = PrimitiveArray::from_iter(vec![width; packed.width_table().len()]);
    let table = bitpack_to_best_bit_width(&table, &mut ctx)?.into_array();
    let table_idx = serialized.children.len() - 2;
    serialized.children[table_idx] = table;
    let buffers: Vec<_> = serialized
        .buffers
        .into_iter()
        .map(BufferHandle::new_host)
        .collect();
    let read = BitPackedPlugin.deserialize(
        ArrayDeserialization::new(
            serialized.serialized_id,
            packed.dtype(),
            packed.len(),
            &serialized.metadata,
            &buffers,
            &StrictChildren(serialized.children),
        ),
        &SESSION,
    )?;
    assert!(read.as_::<BitPacked>().width_table().is::<BitPacked>());
    assert!(read.execute::<PrimitiveArray>(&mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case::widths(true, false)]
#[case::offsets(false, true)]
#[case::both(true, true)]
fn compressed_layout_kernels(
    #[case] compress_widths: bool,
    #[case] compress_offsets: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let packed = encode(&varied(100))?;
    let compress = |child: &ArrayRef, enabled: bool| -> VortexResult<ArrayRef> {
        if !enabled {
            return Ok(child.clone());
        }
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = child.clone().execute::<PrimitiveArray>(&mut ctx)?;
        Ok(bitpack_to_best_bit_width(&primitive, &mut ctx)?.into_array())
    };
    let widths = compress(packed.width_table(), compress_widths)?;
    let offsets = compress(packed.chunk_offsets(), compress_offsets)?;
    let packed = BitPacked::with_chunk_layout(packed, widths, offsets)?;
    let (_, _, read) = serde_roundtrip(&packed)?;
    assert_eq!(
        read.as_::<BitPacked>().width_table().is::<BitPacked>(),
        compress_widths
    );
    assert_eq!(
        read.as_::<BitPacked>().chunk_offsets().is::<BitPacked>(),
        compress_offsets
    );
    assert_arrays_eq!(read, primitive(&varied(100)), &mut ctx);
    let packed = packed.into_array();
    test_array_consistency(&packed, &mut ctx);
    test_take_conformance(&packed, &mut ctx);
    test_filter_conformance(&packed, &mut ctx);
    test_cast_conformance(&packed, &mut ctx);
    test_binary_numeric_array(&packed, &mut ctx);
    assert_arrays_eq!(
        packed.slice(900..2100)?,
        primitive(&varied(100)).slice(900..2100)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn slice_preserves_offset_origin() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072u32).map(|i| match i / 1024 {
        0 => i % 8,
        1 => i % 32,
        _ => i % 4,
    }));
    let packed =
        bitpack_encode_with_widths(&values, ChunkWidths::new(buffer![3u8, 5, 2]), &mut ctx)?
            .into_array();
    let slice = SliceArray::new(packed.clone(), 1100..2300).into_array();
    let sliced = packed
        .reduce_parent(&slice, 0)?
        .ok_or_else(|| vortex_err!("expected bitpacked slice"))?;
    let bp = sliced.as_::<BitPacked>();
    assert_eq!(bp.offset(), 76);
    assert_eq!(bp.packed().len(), 896);
    assert_arrays_eq!(
        bp.chunk_offsets(),
        buffer![384u64, 1024, 1280].into_array(),
        &mut ctx
    );
    assert_arrays_eq!(bp.width_table(), buffer![5u8, 2].into_array(), &mut ctx);
    // Both primitive children and the packed bytes remain views into the original buffers.
    let original = packed.as_::<BitPacked>();
    assert_eq!(
        bp.packed().as_host().as_ptr(),
        original.packed().as_host()[384..].as_ptr()
    );
    assert_eq!(
        bp.chunk_offsets()
            .as_::<Primitive>()
            .as_slice::<u64>()
            .as_ptr(),
        original
            .chunk_offsets()
            .as_::<Primitive>()
            .as_slice::<u64>()[1..]
            .as_ptr()
    );
    let (_, _, read) = serde_roundtrip(&bp.into_owned())?;
    assert_arrays_eq!(
        read,
        values.clone().into_array().slice(1100..2300)?,
        &mut ctx
    );
    let nested = sliced.slice(1000..1200)?;
    assert_arrays_eq!(nested, values.into_array().slice(2100..2300)?, &mut ctx);
    Ok(())
}

#[rstest]
#[case::decreasing(buffer![0u64, 128, 256, 128])]
#[case::wrong_width(buffer![0u64, 128, 256, 385])]
#[case::out_of_bounds(buffer![0u64, 128, 256, u64::MAX])]
fn invalid_offsets_rejected_before_unpacking(#[case] offsets: Buffer<u64>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072u32).map(|i| i % 2));
    let packed = bitpack_to_best_bit_width(&values, &mut ctx)?;
    let widths = packed.width_table().clone();
    let offsets = offsets.into_array();
    assert!(BitPacked::with_chunk_layout(packed.clone(), widths.clone(), offsets.clone()).is_err());
    let offsets = offsets.execute::<PrimitiveArray>(&mut ctx)?;
    let offsets = bitpack_to_best_bit_width(&offsets, &mut ctx)?.into_array();
    let packed = BitPacked::with_chunk_layout(packed, widths, offsets)?.into_array();
    // An isolated scalar checks only its own chunk; bulk unpacking validates the whole layout.
    assert_eq!(packed.execute_scalar(1, &mut ctx)?, Scalar::from(1u32));
    assert!(packed.execute_scalar(2048, &mut ctx).is_err());
    assert!(packed.execute::<PrimitiveArray>(&mut ctx).is_err());
    Ok(())
}

#[test]
fn slice_rejects_unaligned_offsets() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072u32).map(|i| i % 2));
    let packed = bitpack_to_best_bit_width(&values, &mut ctx)?;
    let widths = packed.width_table().clone();
    let offsets = PrimitiveArray::from_iter([0u64, 127, 255, 383]);
    let offsets = bitpack_to_best_bit_width(&offsets, &mut ctx)?.into_array();
    let packed = BitPacked::with_chunk_layout(packed, widths, offsets)?;
    assert!(<BitPacked as SliceKernel>::slice(packed.as_view(), 1024..2048, &mut ctx).is_err());
    Ok(())
}

#[test]
fn offset_child_shape_is_validated() -> VortexResult<()> {
    let packed = encode(&varied(100))?;
    let widths = packed.width_table().clone();
    assert!(
        BitPacked::with_chunk_layout(packed.clone(), widths.clone(), buffer![0u64].into_array())
            .is_err()
    );
    let wrong_dtype =
        PrimitiveArray::from_iter(vec![0u32; packed.chunk_offsets().len()]).into_array();
    assert!(BitPacked::with_chunk_layout(packed, widths, wrong_dtype).is_err());
    Ok(())
}

/// Every array carries a non-nullable `u8` width per chunk, including uniform arrays.
#[test]
fn width_table_is_validated() -> VortexResult<()> {
    let packed = encode(&varied(100))?;
    let num_chunks = packed
        .chunk_widths(&mut SESSION.create_execution_ctx())?
        .len();
    let short = PrimitiveArray::from_iter(vec![3u8; num_chunks - 1]).into_array();
    assert!(BitPacked::with_width_table(packed.clone(), short).is_err());
    let wide = PrimitiveArray::from_iter(vec![3u16; num_chunks]).into_array();
    assert!(BitPacked::with_width_table(packed, wide).is_err());

    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert!(uniform.width_table().is::<Constant>());
    let table = PrimitiveArray::from_iter(vec![
        7u8;
        uniform
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .len()
    ])
    .into_array();
    let replaced = BitPacked::with_width_table(uniform.clone(), table)?;
    assert_arrays_eq!(uniform, replaced, &mut SESSION.create_execution_ctx());
    Ok(())
}

/// Children that report a dtype or length mismatch as an error, as a file reader does, instead
/// of panicking like the slice implementation.
struct StrictChildren(Vec<ArrayRef>);

impl ArrayChildren for StrictChildren {
    fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
        let child =
            <[ArrayRef]>::get(&self.0, index).ok_or_else(|| vortex_err!("no child {index}"))?;
        vortex_ensure!(
            child.dtype() == dtype,
            "child {index} has dtype {}, expected {dtype}",
            child.dtype()
        );
        vortex_ensure!(
            child.len() == len,
            "child {index} has length {}, expected {len}",
            child.len()
        );
        Ok(child.clone())
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

/// The plugin owns wire children, which differ from in-memory slots for the v1 format.
#[test]
fn bare_vtable_requires_plugin() -> VortexResult<()> {
    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert!(ArrayVTable::serialize(uniform.as_view(), &SESSION).is_err());
    let differing = encode(&varied(100))?;
    assert!(ArrayVTable::serialize(differing.as_view(), &SESSION).is_err());
    Ok(())
}

/// Each ID keeps its contract: the original ID cannot read children that carry a width table,
/// and the v2 ID demands one.
#[test]
fn each_format_keeps_its_contract() -> VortexResult<()> {
    let read_as = |array: &BitPackedArray, id: ArrayId| -> VortexResult<()> {
        let array_ref = array.as_array();
        let serialization = SESSION
            .array_serialize(array_ref)?
            .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
        let children = StrictChildren(serialization.children.clone());
        let buffers = serialization
            .buffers
            .clone()
            .into_iter()
            .map(BufferHandle::new_host)
            .collect::<Vec<_>>();
        ArrayPlugin::deserialize(
            &BitPackedPlugin,
            ArrayDeserialization::new(
                id,
                array_ref.dtype(),
                array_ref.len(),
                &serialization.metadata,
                &buffers,
                &children,
            ),
            &SESSION,
        )
        .map(|_| ())
    };

    let differing = encode(&varied(100))?;
    assert!(
        read_as(&differing, ArrayVTable::id(&BitPacked)).is_err(),
        "the original ID must reject a width table"
    );
    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert!(
        read_as(&uniform, bitpacked_v2_id()).is_err(),
        "the v2 ID must demand a width table"
    );
    Ok(())
}
