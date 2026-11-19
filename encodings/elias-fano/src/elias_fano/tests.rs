// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use num_traits::AsPrimitive;
use prost::Message;
use rstest::rstest;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
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
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::EliasFano;
use super::EliasFanoArray;
use super::EliasFanoArrayExt;
use super::array::EliasFanoArraySlotsExt;
use super::array::EliasFanoSlots;
use super::elias_fano_encode;
use super::vtable::EliasFanoMetadata;
use crate::Unary;
use crate::unary_encode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// `values` as a primitive array of `ptype`, which must hold every one of them.
fn primitive(ptype: PType, values: &[u64]) -> PrimitiveArray {
    match_each_unsigned_integer_ptype!(ptype, |T| {
        PrimitiveArray::from_iter(values.iter().map(|&value| AsPrimitive::<T>::as_(value)))
    })
}

fn encode_as(ptype: PType, values: &[u64]) -> VortexResult<Option<EliasFanoArray>> {
    elias_fano_encode(
        primitive(ptype, values).as_ref().as_::<Primitive>(),
        &mut SESSION.create_execution_ctx(),
    )
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

#[test]
fn test_layout_example() -> VortexResult<()> {
    let array = encode_as(PType::U64, &[3, 4, 9])?
        .ok_or_else(|| vortex_err!("the example did not encode"))?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(array.lower_bits, 1);
    // No count exceeds 1, so `upper` stores them as `u8` under its `u64` dtype.
    assert_eq!(
        array.upper().as_::<Widen>().values().dtype(),
        &DType::Primitive(PType::U8, NonNullable)
    );
    assert_arrays_eq!(
        array.upper().clone(),
        PrimitiveArray::from_iter([0u64, 1, 1, 0, 1]),
        &mut ctx
    );
    assert_arrays_eq!(
        array.lower().clone(),
        PrimitiveArray::from_iter([1u64, 0, 1]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::empty(PType::U64, vec![])]
#[case::single(PType::U64, vec![7])]
#[case::duplicates(PType::U32, vec![2, 2, 2, 5, 5, 9])]
#[case::zeros(PType::U16, vec![0; 100])]
#[case::u8_max(PType::U8, vec![0, 1, 255, 255])]
#[case::u64_max(PType::U64, vec![0, u64::MAX])]
#[case::squares(PType::U64, (0..1000).map(|i| i * i).collect())]
fn test_roundtrip(#[case] ptype: PType, #[case] values: Vec<u64>) -> VortexResult<()> {
    let expected = primitive(ptype, &values);
    let ef = encode_as(ptype, &values)?
        .ok_or_else(|| vortex_err!("{} values did not encode", values.len()))?;
    let encoded = ef.clone().into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(encoded, expected, &mut ctx);
    // `Widen` has no wire form, so the counts are written as a `u64` Primitive.
    let counts = ef.upper().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let writable =
        EliasFano::try_new(counts.into_array(), ef.lower().clone(), ef.lower_bits())?.into_array();
    let decoded = serde_roundtrip(&writable)?;
    assert!(decoded.is::<EliasFano>());
    assert_arrays_eq!(decoded, expected, &mut ctx);
    test_array_consistency(&encoded, &mut ctx);
    Ok(())
}

/// With `upper` in Unary, as the EF scheme stores it, which decoding and `scalar_at` read from its
/// bits.
#[rstest]
#[case::empty(PType::U64, vec![])]
#[case::example(PType::U64, vec![3, 4, 9])]
#[case::duplicates(PType::U64, vec![2, 2, 2, 5, 5, 9])]
#[case::u8_max(PType::U8, vec![0, 1, 255, 255])]
#[case::u64_max(PType::U64, vec![0, u64::MAX])]
#[case::squares(PType::U64, (0..1000).map(|i| i * i).collect())]
fn test_unary_upper(#[case] ptype: PType, #[case] values: Vec<u64>) -> VortexResult<()> {
    let expected = primitive(ptype, &values);
    let ef = encode_as(ptype, &values)?
        .ok_or_else(|| vortex_err!("{} values did not encode", values.len()))?;
    let mut ctx = SESSION.create_execution_ctx();
    let counts = ef.upper().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let upper = unary_encode(counts.as_ref().as_::<Primitive>(), &mut ctx)?
        .ok_or_else(|| vortex_err!("the counts did not encode"))?;
    let array = EliasFano::try_new(upper.into_array(), ef.lower().clone(), ef.lower_bits())?;
    // The bits are decoded as they are, without executing `upper` first.
    assert!(matches!(
        EliasFano::execute(array.clone(), &mut ctx)?.step(),
        ExecutionStep::Done
    ));
    let array = array.into_array();
    assert_arrays_eq!(array, expected, &mut ctx);
    // It reads back with `upper` still in Unary.
    let decoded = serde_roundtrip(&array)?;
    assert!(decoded.as_::<EliasFano>().upper().is::<Unary>());
    assert_arrays_eq!(decoded, expected, &mut ctx);
    test_array_consistency(&array, &mut ctx);
    Ok(())
}

#[test]
fn test_encode_returns_none() -> VortexResult<()> {
    let inputs = [
        PrimitiveArray::new(buffer![1u64, 2, 3], Validity::AllValid),
        PrimitiveArray::from_iter([1i64, 2, 3]),
        PrimitiveArray::from_iter([1u64, 3, 2]),
    ];
    for input in inputs {
        assert!(
            elias_fano_encode(
                input.as_ref().as_::<Primitive>(),
                &mut SESSION.create_execution_ctx()
            )?
            .is_none(),
            "{} encoded",
            input.dtype()
        );
    }
    Ok(())
}

#[test]
fn test_rejects() -> VortexResult<()> {
    let u64s = |values: &[u64]| primitive(PType::U64, values).into_array();
    let rejected = [
        (
            "lower_bits at the bit width",
            EliasFano::try_new(u64s(&[1]), u64s(&[0]), 64),
        ),
        (
            "a u32 upper",
            EliasFano::try_new(primitive(PType::U32, &[1]).into_array(), u64s(&[0]), 0),
        ),
        (
            "a signed dtype",
            EliasFano::try_new(
                u64s(&[1]),
                PrimitiveArray::from_iter([0i64]).into_array(),
                0,
            ),
        ),
        (
            "a nullable dtype",
            EliasFano::try_new(
                u64s(&[1]),
                PrimitiveArray::new(buffer![0u64], Validity::AllValid).into_array(),
                0,
            ),
        ),
        // With 7 of a u8's 8 bits low, only buckets 0 and 1 fit.
        (
            "a bucket past the ptype",
            EliasFano::try_new(u64s(&[0, 0, 1]), primitive(PType::U8, &[0]).into_array(), 7),
        ),
    ];
    for (case, result) in rejected {
        assert!(result.is_err(), "{case} was accepted");
    }

    // An array longer than its `lower` child.
    let array = EliasFano::try_new(u64s(&[1]), u64s(&[0]), 0)?;
    assert!(
        Array::try_from_parts(ArrayParts::new(
            EliasFano,
            array.dtype().clone(),
            2,
            array.data().clone(),
            EliasFanoSlots {
                upper: array.upper().clone(),
                lower: array.lower().clone(),
            }
            .into_slots(),
        ))
        .is_err()
    );
    // One slot short.
    assert!(
        Array::try_from_parts(ArrayParts::new(
            EliasFano,
            array.dtype().clone(),
            1,
            array.data().clone(),
            [Some(array.upper().clone())].into_iter().collect(),
        ))
        .is_err()
    );

    // Children whose values break the layout pass `validate`, so decoding must catch them.
    let undecodable = [
        (
            "counts summing past the length",
            u64s(&[3]),
            u64s(&[0, 0]),
            0,
        ),
        (
            "a low wider than lower_bits",
            u64s(&[1, 1]),
            u64s(&[3, 0]),
            1,
        ),
    ];
    let mut ctx = SESSION.create_execution_ctx();
    for (case, counts, lower, lower_bits) in undecodable {
        // Each case runs on the counts, and on the same counts in Unary, which decodes its bits.
        let unary = unary_encode(counts.as_::<Primitive>(), &mut ctx)?
            .ok_or_else(|| vortex_err!("the counts for {case} did not encode"))?
            .into_array();
        for upper in [counts, unary] {
            let array = EliasFano::try_new(upper, lower.clone(), lower_bits)?.into_array();
            assert!(
                array.execute::<PrimitiveArray>(&mut ctx).is_err(),
                "{case} decoded"
            );
        }
    }
    Ok(())
}

#[test]
fn test_deserialize_rejects() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::U64, NonNullable);
    let metadata = |lower_bits: u32| {
        EliasFanoMetadata {
            lower_bits,
            num_buckets: 1,
        }
        .encode_to_vec()
    };
    let children = [
        primitive(PType::U64, &[1]).into_array(),
        primitive(PType::U64, &[0]).into_array(),
    ];
    let deserialize = |metadata: &[u8], buffers: &[BufferHandle], children: &[ArrayRef]| {
        VTable::deserialize(
            &EliasFano, &dtype, 1, metadata, buffers, &children, &SESSION,
        )
        .and_then(Array::try_from_parts)
    };

    deserialize(&metadata(0), &[], &children)?;
    let buffer = BufferHandle::new_host(buffer![0u8]);
    assert!(deserialize(&metadata(0), &[buffer], &children).is_err());
    assert!(deserialize(&metadata(0), &[], &children[..1]).is_err());
    // Too wide for the `u8` that holds `lower_bits`, where a cast would wrap it to 0.
    assert!(deserialize(&metadata(256), &[], &children).is_err());
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_elias_fano_metadata() {
    check_metadata(
        "elias_fano.metadata",
        &EliasFanoMetadata {
            lower_bits: u32::MAX,
            num_buckets: u64::MAX,
        }
        .encode_to_vec(),
    );
}
