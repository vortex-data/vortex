// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression round trips and regressions involving the built-in schemes.
//!
//! These tests cover scheme composition and configuration. Individual encoding kernels and the
//! generic selection algorithm are tested in their owning crates.

use std::iter;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
#[cfg(feature = "zstd")]
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_fastlanes::Delta;
#[cfg(feature = "zstd")]
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::SESSION;

#[track_caller]
fn assert_roundtrip(
    compressor: &BtrBlocksCompressor,
    input: &ArrayRef,
) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(input, &mut ctx)?;
    assert_arrays_eq!(&compressed, input, &mut ctx);

    Ok(compressed)
}

#[test]
fn test_default_compressor_roundtrip() -> VortexResult<()> {
    let integers = PrimitiveArray::from_option_iter(
        (0..2048i32).map(|i| (i % 7 != 0).then_some(1_000_000 + (i * 37) % 100)),
    );
    let floats = PrimitiveArray::from_iter((0..2048).map(|i| f64::from(i) / 100.0));
    let strings = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("https://example.com/products/{i}"))),
        DType::Utf8(Nullability::Nullable),
    );
    let binary = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("PREFIX_{i:09}").into_bytes())),
        DType::Binary(Nullability::Nullable),
    );
    let decimals = DecimalArray::new(
        (0..2048i64).map(|i| i * 123).collect(),
        DecimalDType::new(12, 2),
        Validity::NonNullable,
    );
    let timestamps = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| 1_700_000_000_000_000 + i * 123_456))
            .into_array(),
        TimeUnit::Microseconds,
        None,
    );
    let elements = StructArray::from_fields(&[
        ("integer", integers.into_array()), //
        ("float", floats.into_array()), //
        ("string", strings.into_array()), //
        ("binary", binary.into_array()), //
        ("decimal", decimals.into_array()), //
        ("timestamp", timestamps.into_array()), //
    ])?;
    let offsets = PrimitiveArray::from_iter((0..=2048i32).step_by(4));
    let input = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    let compressed = assert_roundtrip(&BtrBlocksCompressor::from_session(&SESSION), &input)?;
    assert!(compressed.nbytes() < input.nbytes());

    Ok(())
}

#[rstest]
#[case::unaligned({
    let mut rng = StdRng::seed_from_u64(7);
    let mut value = 500_000i32;
    PrimitiveArray::from_iter((0..1025).map(|_| {
        value += 1 + (rng.next_u32() % 6) as i32;
        value
    }))
})]
#[case::nullable_unaligned(
    PrimitiveArray::from_option_iter(iter::once(None).chain((1i32..=100_000).map(Some)))
)]
fn test_delta_unaligned_roundtrip(#[case] input: PrimitiveArray) -> VortexResult<()> {
    // Zero-padding the final chunk used to inflate the delta span and reject this scheme.
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .build();
    let input = input.into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<Delta>());

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(sum(&compressed, &mut ctx)?, sum(&input, &mut ctx)?);

    Ok(())
}

#[cfg(feature = "zstd")]
#[rstest]
#[case::array_level(vortex_zstd::Zstd.id())]
#[case::buffer_level(vortex_zstd::ZstdBuffers.id())]
fn test_cuda_binary_zstd_follows_editions(#[case] allowed: ArrayId) -> VortexResult<()> {
    let values: Vec<_> = (0..1024u32)
        .map(|i| {
            let mut value = Vec::from(&b"common binary payload prefix "[..]);
            value.extend_from_slice(&i.to_le_bytes());
            value.extend_from_slice(&[b'x'; 96]);
            value
        })
        .collect();
    let input = VarBinViewArray::from_iter_bin(values.iter().map(Vec::as_slice)).into_array();

    // The CUDA preset enables both Zstd schemes; the edition filter must choose between them.
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .only_cuda_compatible()
        .retain_allowed_encodings(&HashSet::from([allowed]))
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert_eq!(compressed.encoding_id(), allowed);

    Ok(())
}
