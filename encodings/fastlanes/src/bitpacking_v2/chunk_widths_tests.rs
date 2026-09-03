// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Behavioural tests for bit-packed arrays whose chunks are packed at different widths.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
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
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::BitPackedV2;
use crate::BitPackedV2Array;
use crate::BitPackedV2ArrayExt;
use crate::ChunkWidths;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::bitpacking_v2::bitpack_compress::bitpack_encode_with_widths;
use crate::bitpacking_v2::bitpack_compress::bitpack_to_best_chunk_widths;
use crate::bitpacking_v2::bitpack_compress::bitpack_to_best_chunk_widths_multipass;

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

fn encode(values: &[u32]) -> VortexResult<BitPackedV2Array> {
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
    if let Some(bp) = sliced.as_opt::<BitPackedV2>() {
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

#[test]
fn serde_roundtrip_keeps_widths() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = varied(100);
    let packed = encode(&values)?;
    let array = packed.as_array();

    let serialization = SESSION.array_serialize(array)?.unwrap();
    let children = array.children();
    let buffers = array
        .buffers()
        .into_iter()
        .map(vortex_array::buffer::BufferHandle::new_host)
        .collect::<Vec<_>>();
    let deserialized = BitPackedV2Array::try_from_parts(ArrayVTable::deserialize(
        &BitPackedV2,
        array.dtype(),
        array.len(),
        &serialization.metadata,
        &buffers,
        &children,
        &SESSION,
    )?)
    .map_err(|_| vortex_err!("expected fastlanes.bitpacked"))?;

    assert_eq!(deserialized.chunk_widths(), packed.chunk_widths());
    assert_arrays_eq!(deserialized, primitive(&values), &mut ctx);
    Ok(())
}

#[rstest]
#[case::varied(encode(&varied(100)).unwrap())]
#[case::varied_exact(encode(&varied(0)).unwrap())]
fn conformance(#[case] array: BitPackedV2Array) {
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
