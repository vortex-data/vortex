// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rewrites that keep an array compressed, and answers that need no decoding.

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

use crate::BLOCK_VALUES;
use crate::CHUNK_VALUES;
use crate::EntropyBins;
use crate::EntropyBinsOptions;
use crate::tests::skewed;

fn encode(values: PrimitiveArray, lag: usize) -> VortexResult<ArrayRef> {
    Ok(EntropyBins::from_primitive(
        values.as_view(),
        8,
        EntropyBinsOptions::new(lag, BLOCK_VALUES),
    )?
    .into_array())
}

#[rstest]
#[case::i32_to_i64(PType::I32, PType::I64, 0)]
#[case::i16_to_i32_lag2(PType::I16, PType::I32, 2)]
#[case::i8_to_i64_lag1(PType::I8, PType::I64, 1)]
#[case::u8_to_u64(PType::U8, PType::U64, 0)]
#[case::u16_to_u32_lag1(PType::U16, PType::U32, 1)]
fn widening_cast_stays_compressed(
    #[case] from: PType,
    #[case] to: PType,
    #[case] lag: usize,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // In range for every integer type, and negative only where the type is signed.
    let shift = if from.is_signed_int() { 60 } else { 0 };
    let values = PrimitiveArray::from_iter(
        skewed(3000, 5)
            .into_iter()
            .map(|v| v.rem_euclid(120) - shift),
    )
    .into_array()
    .cast(DType::Primitive(from, Nullability::NonNullable))?
    .execute::<PrimitiveArray>(&mut ctx)?;
    let target = DType::Primitive(to, Nullability::Nullable);
    let expected = values.clone().into_array().cast(target.clone())?;
    let sliced = encode(values, lag)?.slice(700..2900)?;
    let cast = sliced.cast(target)?;
    assert!(cast.is::<EntropyBins>(), "cast decoded the array: {cast}");
    assert_arrays_eq!(cast, expected.slice(700..2900)?, &mut ctx);
    Ok(())
}

#[test]
fn sign_changing_cast_decodes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..2000u32).map(|i| i % 300));
    let target = DType::Primitive(PType::I64, Nullability::NonNullable);
    let cast = encode(values.clone(), 0)?.cast(target.clone())?;
    assert!(!cast.is::<EntropyBins>());
    assert_arrays_eq!(cast, values.into_array().cast(target)?, &mut ctx);
    Ok(())
}

#[test]
fn mask_stays_compressed() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_option_iter(
        skewed(3000, 6)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i % 5 != 0).then_some(v)),
    );
    let keep = BoolArray::from_iter((0..2000).map(|i| i % 3 != 1)).into_array();
    let masked = encode(values.clone(), 1)?
        .slice(500..2500)?
        .mask(keep.clone())?;
    assert!(
        masked.is::<EntropyBins>(),
        "mask decoded the array: {masked}"
    );
    let expected = values.into_array().slice(500..2500)?.mask(keep)?;
    assert_arrays_eq!(masked, expected, &mut ctx);
    Ok(())
}

/// Each chunk's bins fall entirely on one side of the constant, so no block is decoded.
#[test]
fn compare_answers_whole_chunks_from_bins() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let session = array_session();
    crate::initialize(&session);
    let values: Vec<i64> = (0..CHUNK_VALUES + 5000)
        .map(|i| {
            let jitter = (i as i64 * 7919) % 100;
            if i < CHUNK_VALUES {
                jitter
            } else {
                1000 + jitter
            }
        })
        .collect();
    let array =
        encode(PrimitiveArray::from_iter(values.clone()), 0)?.slice(100..CHUNK_VALUES + 4000)?;
    let n = array.len();
    for (c, op) in [
        (500i64, Operator::Lt),
        (500, Operator::Gt),
        (-1, Operator::Gte),
        (2000, Operator::Eq),
    ] {
        let got = array
            .binary(ConstantArray::new(c, n).into_array(), op)?
            .execute::<BoolArray>(&mut session.create_execution_ctx())?;
        let expected = PrimitiveArray::from_iter(values[100..CHUNK_VALUES + 4000].iter().copied())
            .into_array()
            .binary(ConstantArray::new(c, n).into_array(), op)?;
        assert_arrays_eq!(got.into_array(), expected, &mut ctx);
    }
    Ok(())
}
