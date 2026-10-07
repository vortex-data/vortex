// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Delta roundtrips and compute operations against primitive arrays.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::Delta;
use crate::DeltaArray;
use crate::DeltaArraySlotsExt;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_compress::bitpack_encode;

fn session() -> VortexSession {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
}

fn check_roundtrip(source: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<DeltaArray> {
    let delta = Delta::try_from_primitive_array(source, ctx)?;
    assert!(matches!(delta.bases().validity()?, Validity::NonNullable));
    assert!(matches!(delta.deltas().validity()?, Validity::NonNullable));
    assert!(!delta.bases().dtype().is_nullable());
    assert!(!delta.deltas().dtype().is_nullable());
    if let Some(validity) = delta.validity_child() {
        assert_eq!(validity.len(), source.len());
        assert_eq!(validity.dtype(), &Validity::DTYPE);
    }
    let decoded = delta.as_array().clone().execute::<PrimitiveArray>(ctx)?;
    assert_arrays_eq!(decoded, source, ctx);
    assert_eq!(
        min_max(delta.as_array(), ctx, Default::default())?,
        min_max(source.as_array(), ctx, Default::default())?,
    );

    let valid = source.validity()?.execute_mask(source.len(), ctx)?;
    match_each_integer_ptype!(source.ptype(), |T| {
        check_filled_values(
            source.as_slice::<T>(),
            decoded.to_buffer::<T>(),
            &valid,
            ctx,
        );
    });
    Ok(delta)
}

fn check_filled_values<T: NativePType>(
    values: &[T],
    decoded: Buffer<T>,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
) {
    let mut filled = Vec::with_capacity(values.len());
    let mut previous = T::default();
    for (&value, is_valid) in values.iter().zip(valid.iter()) {
        if is_valid {
            previous = value;
        }
        filled.push(previous);
    }
    assert_arrays_eq!(
        PrimitiveArray::new(decoded, Validity::NonNullable),
        PrimitiveArray::from_iter(filled),
        ctx
    );
}

#[rstest]
#[case::all_null(0)]
#[case::alternating(1)]
#[case::leading_nulls(2)]
fn delta_null_patterns(#[case] pattern: usize) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    for len in [0, 1, 63, 1023, 1024, 1025, 2049, 3072] {
        let validity = Validity::from_iter((0..len).map(|index| match pattern {
            0 => false,
            1 => index % 2 == 1,
            _ => index % FL_CHUNK_SIZE >= 63,
        }));
        let source = PrimitiveArray::new(
            (0..len)
                .map(|index| 1000 + index as u32)
                .collect::<Buffer<_>>(),
            validity,
        );
        let delta = check_roundtrip(&source, &mut ctx)?;
        let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let packed = Delta::try_new(
            delta.bases().clone(),
            bitpack_encode(&deltas, 1, None, &mut ctx)?.into_array(),
            delta.validity()?,
            0,
            len,
        )?;
        assert_arrays_eq!(packed, source, &mut ctx);

        for array in [delta.into_array(), packed.into_array()] {
            let write_ctx = ArrayContext::empty();
            let mut bytes = ByteBufferMut::empty();
            for buffer in array.serialize(&write_ctx, &session, &SerializeOptions::default())? {
                bytes.extend_from_slice(&buffer);
            }
            let restored = SerializedArray::try_from(bytes.freeze())?.decode(
                array.dtype(),
                array.len(),
                &ReadContext::new(write_ctx.to_ids()),
                &session,
            )?;
            let restored_delta = restored
                .as_opt::<Delta>()
                .ok_or_else(|| vortex_err!("expected deserialized Delta"))?;
            assert!(!restored_delta.deltas().dtype().is_nullable());
            assert_arrays_eq!(restored, source, &mut ctx);
        }
    }
    Ok(())
}

#[rstest]
#[case::u8(PType::U8)]
#[case::u16(PType::U16)]
#[case::u32(PType::U32)]
#[case::u64(PType::U64)]
#[case::i8(PType::I8)]
#[case::i16(PType::I16)]
#[case::i32(PType::I32)]
#[case::i64(PType::I64)]
fn delta_randomized_model(#[case] ptype: PType) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0xde17a);
    for _ in 0..64 {
        let len = rng.random_range(1..=3073);
        let valid_probability = rng.random_range(0.0..=1.0);
        let validity = Validity::from_iter((0..len).map(|_| rng.random_bool(valid_probability)));
        let source = match_each_integer_ptype!(ptype, |T| {
            PrimitiveArray::new(
                (0..len).map(|_| rng.random::<T>()).collect::<Buffer<_>>(),
                validity,
            )
        });
        let delta = check_roundtrip(&source, &mut ctx)?.into_array();
        let start = rng.random_range(0..=len);
        let end = rng.random_range(start..=len);
        let sliced = delta.slice(start..end)?;
        assert_arrays_eq!(sliced, source.slice(start..end)?, &mut ctx);
        let nested_start = rng.random_range(0..=end - start);
        assert_arrays_eq!(
            sliced.slice(nested_start..end - start)?,
            source.slice(start + nested_start..end)?,
            &mut ctx
        );

        let indices = PrimitiveArray::from_option_iter((0..32).map(|_| {
            rng.random_bool(0.8)
                .then(|| rng.random_range(0..len) as u64)
        }))
        .into_array();
        assert_arrays_eq!(
            delta.take(indices.clone())?,
            source.take(indices)?,
            &mut ctx
        );
        let mask = Mask::from_iter((0..len).map(|_| rng.random_bool(0.5)));
        assert_arrays_eq!(delta.filter(mask.clone())?, source.filter(mask)?, &mut ctx);
    }
    Ok(())
}

#[test]
fn delta_stats_include_null_slot_residuals() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_option_iter(
        (0u32..1024).map(|index| (index % 2 == 1).then_some(1000 + index)),
    );
    let delta = check_roundtrip(&source, &mut ctx)?;
    let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(deltas.statistics().compute_min::<u32>(&mut ctx), Some(0));
    assert_eq!(deltas.statistics().compute_max::<u32>(&mut ctx), Some(1001));
    assert_eq!(delta.statistics().compute_min::<u32>(&mut ctx), Some(1001));
    assert_eq!(delta.statistics().compute_max::<u32>(&mut ctx), Some(2023));
    Ok(())
}

#[rstest]
#[case::all_valid(Validity::AllValid)]
#[case::all_invalid(Validity::AllInvalid)]
#[case::bitmap(Validity::from_iter((0..FL_CHUNK_SIZE).map(|index| index % 2 == 0)))]
fn delta_rejects_nullable_deltas(#[case] validity: Validity) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_iter(0u32..1024);
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let nullable = PrimitiveArray::new(deltas.to_buffer::<u32>(), validity);
    assert!(
        Delta::try_new(
            delta.bases().clone(),
            nullable.into_array(),
            Validity::AllValid,
            0,
            source.len()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn delta_rejects_invalid_top_validity() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_iter(0u32..100);
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    for validity in [
        BoolArray::from_iter([true, false]).into_array(),
        PrimitiveArray::from_iter(0u32..100).into_array(),
    ] {
        assert!(
            Delta::try_new(
                delta.bases().clone(),
                delta.deltas().clone(),
                Validity::Array(validity),
                0,
                source.len()
            )
            .is_err()
        );
    }
    Ok(())
}
