// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::DefaultHasher;
use std::hash::Hasher;
use std::slice;
use std::sync::LazyLock;

use num_traits::AsPrimitive;
use prost::Message;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rstest::rstest;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::EqMode;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::test_harness::check_metadata;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::Unary;
use super::UnaryArray;
use super::array::UnaryMetadata;
use super::bits::for_each_unset_index;
use super::unary_encode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// `values` as a primitive array of `ptype`, which must hold every one of them.
fn primitive(ptype: PType, values: &[u64]) -> PrimitiveArray {
    match_each_unsigned_integer_ptype!(ptype, |T| {
        PrimitiveArray::from_iter(values.iter().map(|&value| AsPrimitive::<T>::as_(value)))
    })
}

fn encode_as(ptype: PType, values: &[u64]) -> VortexResult<Option<UnaryArray>> {
    let array = primitive(ptype, values);
    unary_encode(array.as_ref().as_::<Primitive>())
}

fn encode(values: &[u64]) -> VortexResult<UnaryArray> {
    encode_as(PType::U64, values)?
        .ok_or_else(|| vortex_err!("{} values did not encode", values.len()))
}

/// `len` values drawn from `0..=max`, in no particular order.
fn small(len: usize, max: u64, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(0..=max)).collect()
}

/// Every value read back through `execute_scalar`, in a shuffled order and then in sequence.
fn check_access(array: &ArrayRef, expected: &PrimitiveArray, seed: u64) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut order: Vec<usize> = (0..array.len()).collect();
    order.shuffle(&mut StdRng::seed_from_u64(seed));
    for index in order.into_iter().chain(0..array.len()) {
        assert_eq!(
            array.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(index, &mut ctx)?,
            "index {index}"
        );
    }
    Ok(())
}

fn serde_roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let mut concat = ByteBufferMut::empty();
    for buffer in array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        concat.extend_from_slice(buffer.as_ref());
    }
    SerializedArray::try_from(concat.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}

fn hash_of(array: &ArrayRef) -> u64 {
    let mut hasher = DefaultHasher::new();
    array.array_hash(&mut hasher, EqMode::Value);
    hasher.finish()
}

#[test]
fn test_layout_example() -> VortexResult<()> {
    let array = encode(&[2, 0, 3])?;
    assert_eq!(
        array.bits(),
        &BitBuffer::from(vec![true, true, false, false, true, true, true, false])
    );
    Ok(())
}

#[rstest]
#[case::empty(PType::U64, vec![])]
#[case::single(PType::U64, vec![42])]
#[case::all_zeros(PType::U64, vec![0; 300])]
#[case::all_equal(PType::U64, vec![7; 300])]
#[case::unsorted(PType::U64, small(500, 3, 1))]
#[case::decreasing(PType::U64, (0..60).rev().collect())]
#[case::runs(PType::U64, (0..500).map(|i| i / 64 * 3).collect())]
#[case::u8(PType::U8, small(300, 7, 2))]
#[case::u8_max(PType::U8, (0..41).map(|i| if i == 20 { 255 } else { 0 }).collect())]
#[case::u16(PType::U16, small(500, 20, 3))]
#[case::u32(PType::U32, small(500, 40, 4))]
// A value of 1,000 that starts 4 bits into a byte, so reading it scans whole 512-bit chunks from an
// offset.
#[case::long_value(PType::U16, [vec![3, 1000], vec![0; 68]].concat())]
fn test_roundtrip(#[case] ptype: PType, #[case] values: Vec<u64>) -> VortexResult<()> {
    let expected = primitive(ptype, &values);
    let encoded = encode_as(ptype, &values)?
        .ok_or_else(|| vortex_err!("{} values did not encode", values.len()))?
        .into_array();
    assert_eq!(encoded.dtype(), &DType::Primitive(ptype, NonNullable));
    assert_arrays_eq!(encoded, expected, &mut SESSION.create_execution_ctx());
    check_access(&encoded, &expected, 0x5EED)
}

#[test]
fn test_encode_returns_none() -> VortexResult<()> {
    let inputs = [
        PrimitiveArray::new(buffer![1u64, 2, 3], Validity::AllValid),
        PrimitiveArray::from_iter([1i64, 2, 3]),
        PrimitiveArray::from_iter([1.0f64, 2.0, 3.0]),
        // 2 + 1,000 bits, against the plain array's 128.
        PrimitiveArray::from_iter([0u64, 1000]),
        // The sum overflows a `u64`.
        PrimitiveArray::from_iter([u64::MAX, 1]),
    ];
    for input in inputs {
        assert!(
            unary_encode(input.as_ref().as_::<Primitive>())?.is_none(),
            "{} encoded",
            input.dtype()
        );
    }
    Ok(())
}

#[rstest]
#[case(PType::U8)]
#[case(PType::U16)]
#[case(PType::U32)]
#[case(PType::U64)]
fn test_size_rule_boundary(#[case] ptype: PType) -> VortexResult<()> {
    // Four values take `4 + num_ones` bits, against the plain array's `4 × width`.
    let num_ones = 4 * u64::try_from(ptype.bit_width())? - 4;
    assert!(encode_as(ptype, &[0, 0, 0, num_ones])?.is_some());
    assert!(encode_as(ptype, &[0, 0, 0, num_ones + 1])?.is_none());
    Ok(())
}

#[test]
fn test_try_new_rejects() -> VortexResult<()> {
    let bits = |bits: &[bool]| BitBuffer::from(bits);
    let rejected = [
        ("signed", Unary::try_new(bits(&[false]), PType::I64, 1)),
        (
            "nonzero offset",
            Unary::try_new(bits(&[true, true, true, false]).slice(3..), PType::U64, 1),
        ),
        (
            "fewer bits than values",
            Unary::try_new(bits(&[false, false]), PType::U64, 3),
        ),
        (
            "last bit set",
            Unary::try_new(bits(&[false, true]), PType::U64, 1),
        ),
        (
            "empty with bits",
            Unary::try_new(bits(&[true]), PType::U64, 0),
        ),
        // `[1, 0]` takes the bits `10 0`. Clearing bit 0 or setting bit 1 keeps the last bit unset,
        // so only the count of set bits catches them.
        (
            "a set bit missing",
            Unary::try_new(bits(&[false, false, false]), PType::U64, 2),
        ),
        (
            "an extra set bit",
            Unary::try_new(bits(&[true, true, false]), PType::U64, 2),
        ),
        // 256 set bits and one unset: a single value of 256.
        (
            "a value above the ptype",
            Unary::try_new((0..257).map(|bit| bit < 256).collect(), PType::U8, 1),
        ),
    ];
    for (case, result) in rejected {
        assert!(result.is_err(), "{case} was accepted");
    }

    // The vtable refuses children however the parts are put together.
    let array = encode(&[1, 2])?;
    let child = primitive(PType::U64, &[1]).into_array();
    assert!(
        Array::try_from_parts(ArrayParts::new(
            Unary,
            array.dtype().clone(),
            array.len(),
            array.data().clone(),
            [Some(child)].into_iter().collect(),
        ))
        .is_err()
    );
    Ok(())
}

#[test]
fn test_deserialize_rejects() {
    let dtype = DType::Primitive(PType::U64, NonNullable);
    let metadata = |num_ones: u64| UnaryMetadata { num_ones }.encode_to_vec();
    // `[1, 0]` takes the bits `10 0`.
    let bits = BufferHandle::new_host(BitBuffer::from(vec![true, false, false]).inner().clone());
    let no_children: [ArrayRef; 0] = [];
    // `try_from_parts` validates the parts, as the array plugin does after `deserialize`.
    let deserialize = |metadata: &[u8], buffers: &[BufferHandle], children: &[ArrayRef]| {
        Unary
            .deserialize(&dtype, 2, metadata, buffers, &children, &SESSION)
            .and_then(Array::try_from_parts)
    };

    let one = slice::from_ref(&bits);

    assert!(deserialize(&metadata(1), one, &no_children).is_ok());
    assert!(deserialize(&metadata(1), &[], &no_children).is_err());
    assert!(deserialize(&metadata(1), &[bits.clone(), bits.clone()], &no_children).is_err());
    assert!(
        deserialize(
            &metadata(1),
            one,
            &[primitive(PType::U64, &[1]).into_array()]
        )
        .is_err()
    );
    // The buffer holds 8 bits, short of the 102 these claim.
    assert!(deserialize(&metadata(100), one, &no_children).is_err());
    // 2 + `u64::MAX` bits overflow the count.
    assert!(deserialize(&metadata(u64::MAX), one, &no_children).is_err());
    // `11 0` keeps the last bit unset, but holds two set bits where the metadata says one.
    let extra = BufferHandle::new_host(BitBuffer::from(vec![true, true, false]).inner().clone());
    assert!(deserialize(&metadata(1), slice::from_ref(&extra), &no_children).is_err());
}

#[test]
fn test_serde_roundtrip() -> VortexResult<()> {
    let values = small(1000, 7, 6);
    let decoded = serde_roundtrip(&encode(&values)?.into_array())?;
    assert!(decoded.is::<Unary>());
    assert_arrays_eq!(
        decoded,
        primitive(PType::U64, &values),
        &mut SESSION.create_execution_ctx()
    );
    Ok(())
}

#[test]
fn test_with_buffers_roundtrip() -> VortexResult<()> {
    let array = encode(&small(1000, 7, 7))?.into_array();
    let buffers = array.buffer_handles();
    // SAFETY: the replacement buffers are the array's own, so they hold the same values.
    let rebuilt = unsafe { array.clone().with_buffers(buffers) }?;
    assert!(rebuilt.is::<Unary>());
    assert!(rebuilt.array_eq(&array, EqMode::Value));
    Ok(())
}

#[test]
fn test_array_eq_hash() -> VortexResult<()> {
    let array = encode(&[1, 0, 2])?.into_array();
    let same = encode(&[1, 0, 2])?.into_array();
    assert!(array.array_eq(&same, EqMode::Value));
    assert_eq!(hash_of(&array), hash_of(&same));

    // `[0, 1, 2]` has the same length and sum, with different bits.
    for other in [encode(&[0, 1, 2])?, encode(&[1, 0, 3])?] {
        assert!(!array.array_eq(&other.into_array(), EqMode::Value));
    }
    Ok(())
}

#[test]
fn test_nbytes() -> VortexResult<()> {
    let values = small(1000, 7, 9);
    let num_ones: u64 = values.iter().sum();
    assert_eq!(
        encode(&values)?.into_array().nbytes(),
        (1000 + num_ones).div_ceil(8)
    );

    // The value 2, as the first 3 bits of a 125-byte buffer, keeps only the byte it uses.
    let prefix: BitBuffer = (0..1000).map(|bit| bit < 2).collect();
    assert_eq!(
        Unary::try_new(prefix.slice(..3), PType::U64, 1)?
            .into_array()
            .nbytes(),
        1
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_unary_metadata() {
    check_metadata(
        "unary.metadata",
        &UnaryMetadata { num_ones: u64::MAX }.encode_to_vec(),
    );
}

#[rstest]
#[case::small(small(1000, 3, 10))]
#[case::wide(small(1000, 40, 11))]
#[case::runs((0..1000).map(|i| i / 64).collect())]
#[case::zeros(vec![0; 100])]
#[case::single(vec![5])]
fn test_unary_consistency(#[case] values: Vec<u64>) -> VortexResult<()> {
    let array = encode(&values)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    test_array_consistency(&array, &mut ctx);
    if values.len() > 2 {
        test_array_consistency(&array.slice(1..values.len() - 1)?, &mut ctx);
    }
    Ok(())
}

#[test]
fn test_lazy_slice() -> VortexResult<()> {
    let values = small(1000, 7, 13);
    let sliced = encode(&values)?.into_array().slice(100..900)?;
    let expected = primitive(PType::U64, &values[100..900]);
    assert_arrays_eq!(sliced, expected, &mut SESSION.create_execution_ctx());
    check_access(&sliced, &expected, 0x51CE)
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(63)]
#[case(64)]
#[case(65)]
#[case(128)]
#[case(200)]
fn test_for_each_unset_index_agrees_with_naive(#[case] len: usize) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(u64::try_from(len)?);
    let random: BitBuffer = (0..len).map(|_| rng.random_bool(0.5)).collect();
    // The same kind of bits, starting 3 bits into their buffer.
    let offset: BitBuffer = (0..len + 3).map(|_| rng.random_bool(0.5)).collect();
    for bits in [
        random,
        offset.slice(3..),
        BitBuffer::new_set(len),
        BitBuffer::new_unset(len),
    ] {
        let expected: Vec<usize> = bits
            .iter()
            .enumerate()
            .filter_map(|(index, set)| (!set).then_some(index))
            .collect();
        let mut found = Vec::new();
        for_each_unset_index(&bits, |index| found.push(index));
        assert_eq!(found, expected);
    }
    Ok(())
}
