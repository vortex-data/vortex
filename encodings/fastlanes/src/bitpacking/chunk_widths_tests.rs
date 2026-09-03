// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Behavioural tests for bit-packed arrays whose chunks are packed at different widths.

use std::sync::LazyLock;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
use vortex_array::compute::conformance::cast::test_cast_conformance;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::compute::conformance::filter::test_filter_conformance;
use vortex_array::compute::conformance::take::test_take_conformance;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::ArrayChildren;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::BitPackedPlugin;
use crate::ChunkWidths;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::bitpacked_v2_id;
use crate::bitpacking::bitpack_compress::bitpack_encode_with_widths;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;
use crate::bitpacking::bitpack_compress::bitpack_to_best_chunk_widths;
use crate::bitpacking::bitpack_compress::bitpack_to_best_chunk_widths_multipass;
use crate::bitpacking::plugin::BitPackedV2Metadata;
use crate::bitpacking::vtable::BitPackedMetadata;

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
    bitpack_to_best_chunk_widths(&PrimitiveArray::from_iter(values.iter().copied()), &mut ctx)
}

fn primitive(values: &[u32]) -> ArrayRef {
    PrimitiveArray::from_iter(values.iter().copied()).into_array()
}

#[test]
fn picks_a_width_per_chunk() -> VortexResult<()> {
    let packed = encode(&varied(100))?;
    let widths = packed.chunk_widths();
    assert_eq!(
        widths.uniform_width(),
        None,
        "chunks differ in magnitude: {widths}"
    );
    assert_eq!(widths.len(), 5);
    assert_eq!(widths.width(2), 0, "an all-zero chunk stores nothing");
    assert!(widths.width(0) < widths.width(1));
    assert!(widths.width(1) < widths.width(3));
    assert_eq!(widths.max_width(), widths.width(3));
    assert_eq!(packed.bit_width(), widths.max_width());
    assert!(
        packed.patches().is_some(),
        "chunk 1 outliers become patches"
    );
    Ok(())
}

#[test]
fn uniform_data_gets_equal_widths() -> VortexResult<()> {
    let values: Vec<u32> = (0..3000).map(|i| i % 128).collect();
    let packed = encode(&values)?;
    assert_eq!(packed.chunk_widths().len(), 3);
    assert_eq!(packed.chunk_widths().uniform_width(), Some(7));
    assert_eq!(packed.bit_width(), 7);
    Ok(())
}

#[rstest]
#[case::exact_chunks(0)]
#[case::partial_tail(100)]
#[case::single_tail(1)]
fn roundtrip(#[case] tail: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(tail);
    let packed = encode(&values)?;
    assert_arrays_eq!(packed, primitive(&values), &mut ctx);
    Ok(())
}

#[test]
fn scalar_at_every_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    for idx in [
        0,
        5,
        1024,
        1024 + 7,
        1024 + 307,
        2048,
        2500,
        3072,
        4000,
        4095,
        4096,
        4195,
    ] {
        assert_eq!(
            packed.execute_scalar(idx, &mut ctx)?,
            Scalar::from(values[idx]),
            "index {idx}"
        );
    }
    Ok(())
}

#[rstest]
#[case::within_first_chunk(10..900)]
#[case::across_first_boundary(900..1100)]
#[case::whole_middle_chunks(1024..3072)]
#[case::through_zero_chunk(1500..2600)]
#[case::into_tail(3000..4150)]
#[case::tail_only(4100..4196)]
fn slice_matches_primitive(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    let sliced = packed.slice(range.clone())?;
    let expected = primitive(&values).slice(range.clone())?;
    assert_arrays_eq!(sliced, expected, &mut ctx);

    // The slice is still bit-packed and keeps only the widths of the chunks it overlaps.
    let sliced = sliced.execute::<ArrayRef>(&mut ctx)?;
    if let Some(bp) = sliced.as_opt::<BitPacked>() {
        let expected_chunks = (range.end).div_ceil(FL_CHUNK_SIZE) - range.start / FL_CHUNK_SIZE;
        assert_eq!(bp.chunk_widths().len(), expected_chunks);
    }
    assert_eq!(
        sliced.execute_scalar(0, &mut ctx)?,
        Scalar::from(values[range.start])
    );
    Ok(())
}

#[test]
fn take_sparse_indices() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    // Few enough indices that the kernel unpacks single values rather than the whole array.
    let indices = [3usize, 1030, 1031, 1331, 2100, 3500, 4100];
    let taken = packed
        .take(buffer![3u64, 1030, 1031, 1331, 2100, 3500, 4100].into_array())?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_iter(indices.iter().map(|&i| values[i])),
        &mut ctx
    );
    Ok(())
}

#[test]
fn filter_sparse_mask() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    let indices = vec![3usize, 1030, 1031, 1331, 2100, 3500, 4100];
    let filtered = packed
        .filter(Mask::from_indices(values.len(), indices.clone()))?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_arrays_eq!(
        filtered,
        PrimitiveArray::from_iter(indices.iter().map(|&i| values[i])),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case(Operator::Eq)]
#[case(Operator::Lt)]
#[case(Operator::Gte)]
fn compare_constant_matches_primitive(#[case] op: Operator) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    // Zero lands inside the all-zero chunk's range, exercising the zero-width fused path.
    for rhs in [0u32, 5, 3000] {
        let rhs = ConstantArray::new(rhs, values.len()).into_array();
        let got = packed
            .clone()
            .binary(rhs.clone(), op)?
            .execute::<BoolArray>(&mut ctx)?;
        let want = primitive(&values)
            .binary(rhs, op)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(got, want, &mut ctx);
    }
    Ok(())
}

#[test]
fn between_matches_primitive() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    let lower = ConstantArray::new(2u32, values.len()).into_array();
    let upper = ConstantArray::new(3000u32, values.len()).into_array();
    let options = BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::Strict,
    };
    let got = packed
        .between(lower.clone(), upper.clone(), options.clone())?
        .execute::<BoolArray>(&mut ctx)?;
    let want = primitive(&values)
        .between(lower, upper, options)?
        .execute::<BoolArray>(&mut ctx)?;
    assert_arrays_eq!(got, want, &mut ctx);
    Ok(())
}

#[test]
fn widening_cast_matches_primitive() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    let target = DType::Primitive(PType::U64, Nullability::NonNullable);
    let got = packed
        .cast(target.clone())?
        .execute::<PrimitiveArray>(&mut ctx)?;
    let want = primitive(&values)
        .cast(target)?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_arrays_eq!(got, want, &mut ctx);
    Ok(())
}

#[test]
fn not_constant() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let packed = encode(&varied(100))?.into_array();
    assert!(!is_constant(&packed, &mut ctx)?);
    Ok(())
}

#[test]
fn nullable_and_signed_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<i32> = varied(50).into_iter().map(|v| v as i32).collect();
    let validity = Validity::from_iter((0..values.len()).map(|i| i % 7 != 0));
    let array = PrimitiveArray::new(Buffer::from_iter(values.iter().copied()), validity.clone());
    let packed = bitpack_to_best_chunk_widths(&array, &mut ctx)?;
    assert_eq!(packed.chunk_widths().uniform_width(), None);
    assert_eq!(
        packed.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    assert_arrays_eq!(
        packed,
        PrimitiveArray::new(Buffer::from_iter(values.iter().copied()), validity),
        &mut ctx
    );
    Ok(())
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
    assert_eq!(packed.bit_width(), 16);
    assert_arrays_eq!(packed, array, &mut ctx);
    Ok(())
}

#[test]
fn for_fused_decode() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?.into_array();
    let for_array = FoR::try_new(packed, Scalar::from(1000u32))?;
    assert_arrays_eq!(
        for_array,
        PrimitiveArray::from_iter(values.iter().map(|v| v + 1000)),
        &mut ctx
    );
    Ok(())
}

/// Serialize `array` through the session and read it back through the plugin registered for the
/// serialized ID, as a file reader would.
fn serde_roundtrip(array: &BitPackedArray) -> VortexResult<(ArrayId, Vec<u8>, ArrayRef)> {
    let array_ref = array.as_array();
    let serialization = SESSION
        .array_serialize(array_ref)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    let children = array_ref.children();
    let buffers = array_ref
        .buffers()
        .into_iter()
        .map(BufferHandle::new_host)
        .collect::<Vec<_>>();
    let plugin = SESSION
        .arrays()
        .registry()
        .get(&serialization.serialized_id)
        .ok_or_else(|| vortex_err!("no plugin for {}", serialization.serialized_id))?;
    let parts = ArrayDeserialization::new(
        serialization.serialized_id,
        array_ref.dtype(),
        array_ref.len(),
        &serialization.metadata,
        &buffers,
        &children,
    );
    let read = plugin.deserialize(parts, &SESSION)?;
    Ok((serialization.serialized_id, serialization.metadata, read))
}

/// Differing chunk widths serialize under the v2 ID with the width table as a child, and read
/// back with the same widths.
#[test]
fn differing_widths_serialize_as_v2() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?;
    let table = packed
        .width_table()
        .ok_or_else(|| vortex_err!("differing widths must carry a width table"))?;
    assert_eq!(
        table
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .as_slice::<u8>(),
        packed.chunk_widths().as_slice()
    );
    let (id, metadata, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, bitpacked_v2_id());
    assert_eq!(BitPackedV2Metadata::decode(metadata.as_slice())?.offset, 0);
    assert_eq!(
        read.as_::<BitPacked>().chunk_widths(),
        packed.chunk_widths()
    );
    assert!(read.as_::<BitPacked>().width_table().is_some());
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
    assert!(packed.width_table().is_none());
    assert!(packed.as_array().children().is_empty());
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
    assert!(packed.width_table().is_none());
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
        packed.chunk_widths().as_buffer().clone(),
        Validity::NonNullable,
    );
    let compressed_table = bitpack_to_best_bit_width(&table, &mut ctx)?.into_array();
    assert!(compressed_table.is::<BitPacked>());
    let packed = BitPacked::with_width_table(packed, compressed_table)?;
    let (id, _, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, bitpacked_v2_id());
    let view = read.as_::<BitPacked>();
    assert_eq!(view.chunk_widths(), packed.chunk_widths());
    assert!(
        view.width_table()
            .is_some_and(|table| table.is::<BitPacked>())
    );
    assert_arrays_eq!(read, primitive(&values), &mut ctx);
    Ok(())
}

/// The width table must hold one `u8` per chunk, and only arrays whose chunk widths differ carry
/// one.
#[test]
fn width_table_is_validated() -> VortexResult<()> {
    let packed = encode(&varied(100))?;
    let num_chunks = packed.chunk_widths().len();
    let short = PrimitiveArray::from_iter(vec![3u8; num_chunks - 1]).into_array();
    assert!(BitPacked::with_width_table(packed.clone(), short).is_err());
    let wide = PrimitiveArray::from_iter(vec![3u16; num_chunks]).into_array();
    assert!(BitPacked::with_width_table(packed, wide).is_err());

    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert!(uniform.width_table().is_none());
    let table = PrimitiveArray::from_iter(vec![7u8; uniform.chunk_widths().len()]).into_array();
    assert!(BitPacked::with_width_table(uniform, table).is_err());
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

/// The encoding's own serializer is labelled with the original ID, so it only ever writes arrays
/// that satisfy that ID's contract.
#[test]
fn bare_vtable_only_writes_the_original_format() -> VortexResult<()> {
    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert!(ArrayVTable::serialize(uniform.as_view(), &SESSION)?.is_some());
    let differing = encode(&varied(100))?;
    assert!(ArrayVTable::serialize(differing.as_view(), &SESSION)?.is_none());
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
        let children = StrictChildren(array_ref.children());
        let buffers = array_ref
            .buffers()
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

#[rstest]
#[case::varied(encode(&varied(100)).unwrap())]
#[case::varied_exact(encode(&varied(0)).unwrap())]
fn conformance(#[case] array: BitPackedArray) {
    let mut ctx = SESSION.create_execution_ctx();
    let array = array.into_array();
    test_array_consistency(&array, &mut ctx);
    test_take_conformance(&array, &mut ctx);
    test_filter_conformance(&array, &mut ctx);
    test_cast_conformance(&array, &mut ctx);
    test_binary_numeric_array(&array, &mut ctx);
}

/// The fused single-walk encoder must produce exactly what the multi-pass one does.
#[rstest]
#[case::varied(PrimitiveArray::from_iter(varied(100)))]
#[case::varied_exact(PrimitiveArray::from_iter(varied(0)))]
#[case::tiny(PrimitiveArray::from_iter([5u32, 1 << 20, 7]))]
#[case::nullable_signed(PrimitiveArray::new(
    Buffer::from_iter(varied(50).into_iter().map(|v| v as i32)),
    Validity::from_iter((0..4 * FL_CHUNK_SIZE + 50).map(|i| i % 7 != 0)),
))]
#[case::all_null(PrimitiveArray::new(Buffer::from_iter(varied(9)), Validity::AllInvalid))]
#[case::short_and_wide(short_and_wide())]
fn fused_matches_multipass(#[case] array: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let fused = bitpack_to_best_chunk_widths(&array, &mut ctx)?;
    let multipass = bitpack_to_best_chunk_widths_multipass(&array, &mut ctx)?;
    assert_eq!(fused.chunk_widths(), multipass.chunk_widths());
    assert_eq!(fused.packed().as_host(), multipass.packed().as_host());
    assert_eq!(
        fused.patches().map(|p| p.num_patches()),
        multipass.patches().map(|p| p.num_patches())
    );
    assert_eq!(fused.nbytes(), multipass.nbytes());
    assert_arrays_eq!(fused, array, &mut ctx);
    Ok(())
}

/// 200 u8 values needing 7 bits: the single padded block (896 bytes) is larger than the raw
/// array (200 bytes), which an encoder sizing its output by the raw length overflows.
fn short_and_wide() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..200u8).map(|i| i.wrapping_mul(97) % 128))
}

#[test]
fn short_chunk_packs_wider_than_raw() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = short_and_wide();
    let packed = bitpack_to_best_chunk_widths(&array, &mut ctx)?;
    assert_eq!(packed.chunk_widths().as_slice(), &[7]);
    assert!(packed.packed().len() > array.nbytes() as usize);
    assert_arrays_eq!(packed, array, &mut ctx);
    Ok(())
}
