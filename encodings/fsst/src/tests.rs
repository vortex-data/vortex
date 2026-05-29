// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::buffer;
use vortex_mask::Mask;

use crate::FSST;
use crate::fsst_compress;
use crate::fsst_train_compressor;

/// this function is VERY slow on miri, so we only want to run it once
pub(crate) fn build_fsst_array() -> ArrayRef {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.append_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.append_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.append_value(b"Nothing in present history can contradict them");
    let input_array = input_array.finish(DType::Utf8(Nullability::NonNullable));

    let compressor = fsst_train_compressor(&input_array);
    let len = input_array.len();
    let dtype = input_array.dtype().clone();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    fsst_compress(input_array, len, &dtype, &compressor, &mut ctx).into_array()
}

#[test]
fn test_fsst_array_ops() {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // first test the scalar_at values
    let fsst_array = build_fsst_array();
    assert_nth_scalar!(
        fsst_array,
        0,
        "The Greeks never said that the limit could not be overstepped"
    );
    assert_nth_scalar!(
        fsst_array,
        1,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );
    assert_nth_scalar!(
        fsst_array,
        2,
        "Nothing in present history can contradict them"
    );

    // test slice
    let fsst_sliced = fsst_array.slice(1..3).unwrap();
    assert!(fsst_sliced.is::<FSST>());
    assert_eq!(fsst_sliced.len(), 2);
    assert_nth_scalar!(
        fsst_sliced,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );
    assert_nth_scalar!(
        fsst_sliced,
        1,
        "Nothing in present history can contradict them"
    );

    // test take
    let indices = buffer![0, 2].into_array();
    let fsst_taken = fsst_array.take(indices).unwrap();
    assert_eq!(fsst_taken.len(), 2);
    assert_nth_scalar!(
        fsst_taken,
        0,
        "The Greeks never said that the limit could not be overstepped"
    );
    assert_nth_scalar!(
        fsst_taken,
        1,
        "Nothing in present history can contradict them"
    );

    // test filter
    let mask = Mask::from_iter([false, true, true]);

    let fsst_filtered = fsst_array.filter(mask).unwrap();

    assert_eq!(fsst_filtered.len(), 2);
    assert_nth_scalar!(
        fsst_filtered,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down"
    );

    // test to_canonical
    let canonical_array = fsst_array
        .clone()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap()
        .into_array();

    assert_arrays_eq!(fsst_array, canonical_array);
}

// TODO(someone): ideally CI would run this in release mode as well since debug builds make the
// allocation and compression loop substantially slower.
/// Regression for #7833: [`fsst_compress`] must accept inputs whose cumulative compressed
/// bytes exceed [`i32::MAX`]. Before the fix, [`fsst_compress_iter`] hardcoded
/// [`VarBinBuilder<i32>`] for the FSST output and panicked in
/// [`VarBinBuilder::append_value`] once cumulative compressed bytes crossed the boundary.
///
/// The input is built with [`VarBinBuilder<i64>`] so the input itself does not panic, which
/// confirms the overflow is on the FSST output side. After the fix the test must succeed
/// with the row count preserved.
///
/// Allocates ~2.5 GiB for the input and ~2.5 GiB for the FSST output (~5 GiB total), so it
/// is gated to CI runs and skipped when `VORTEX_SKIP_SLOW_TESTS` is set. To run it locally:
///
/// ```text
/// CI=1 cargo test --release -p vortex-fsst fsst_compress_offsets
/// ```
///
/// [`fsst_compress_iter`]: crate::compress::fsst_compress_iter
#[test_with::env(CI)]
#[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
fn fsst_compress_offsets_overflow_i32() {
    // High-entropy ASCII strings sliced from a random pool. FSST is a symbol-table
    // compressor; pseudo-random data with no recurring byte sequences resists compression,
    // so the compressed output stays close to input size and crosses the i32 boundary.
    const STRING_LEN: usize = 64 * 1024;
    const TOTAL_BYTES: usize = (1usize << 31) + (512 << 20); // ~2.5 GiB
    const N: usize = TOTAL_BYTES / STRING_LEN;
    const POOL_LEN: usize = 64 * 1024 * 1024;

    // Printable ASCII alphabet so the result is valid UTF-8.
    const ALPHABET: &[u8; 95] =
        b" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

    let mut rng = StdRng::seed_from_u64(0xC0DE_C011_B711);
    let pool: Vec<u8> = (0..POOL_LEN)
        .map(|_| *ALPHABET.choose(&mut rng).unwrap())
        .collect();

    println!("building large VarBinArray");
    let mut builder = VarBinBuilder::<i64>::with_capacity(N);
    for i in 0..N {
        let off = i.wrapping_mul(31337) % (POOL_LEN - STRING_LEN);
        builder.append_value(&pool[off..off + STRING_LEN]);
    }
    let array = builder.finish(DType::Utf8(Nullability::NonNullable));

    println!("training FSST compressor");
    let compressor = fsst_train_compressor(&array);
    let len = array.len();
    let dtype = array.dtype().clone();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    println!("compressing to FSST");
    let compressed = fsst_compress(array, len, &dtype, &compressor, &mut ctx);
    assert_eq!(compressed.len(), len);
}

// TODO(someone): ideally CI would run this in release mode as well since debug builds make the
// allocation and decompression loop substantially slower.
/// Decode-side companion to [`fsst_compress_offsets_overflow_i32`]. Canonicalizing an FSST array
/// decompresses the whole heap and feeds it to `build_views` with `max_buffer_len = MAX_BUFFER_LEN`
/// (`i32::MAX`). `build_views` has a single-buffer fast path taken only when the decoded heap fits
/// within one buffer; when the heap exceeds `MAX_BUFFER_LEN` it must instead roll over into multiple
/// buffers (resetting the per-buffer offset) so the `u32` view offsets never overflow.
///
/// This builds an FSST array whose *decompressed* size crosses `i32::MAX`, so canonicalization is
/// forced down the rollover path. We assert the row count is preserved, that more than one data
/// buffer was produced (proving the fast path was declined), that no single buffer exceeds
/// `MAX_BUFFER_LEN`, and that values on both sides of the rollover boundary reconstruct exactly.
///
/// Highly compressible bodies keep the FSST array tiny, so peak memory is dominated by the ~2.25 GiB
/// input and the ~2.25 GiB decompressed heap. Gated to CI and skipped when `VORTEX_SKIP_SLOW_TESTS`
/// is set. To run it locally:
///
/// ```text
/// CI=1 cargo test --release -p vortex-fsst fsst_canonicalize_offsets
/// ```
#[test_with::env(CI)]
#[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
fn fsst_canonicalize_offsets_overflow_i32() {
    const STRING_LEN: usize = 64 * 1024;
    // Comfortably past MAX_BUFFER_LEN (`i32::MAX` ~= 2.0 GiB) so the decoded heap must roll over.
    const TOTAL_BYTES: usize = (1usize << 31) + (256 << 20); // ~2.25 GiB
    const N: usize = TOTAL_BYTES / STRING_LEN;

    // Each value is a long, trivially compressible body carrying its row index as an ASCII prefix,
    // so FSST compresses the heap to almost nothing while every row stays individually verifiable.
    fn nth_string(i: usize) -> Vec<u8> {
        let mut s = vec![b'x'; STRING_LEN];
        let prefix = format!("row-{i:08}-");
        s[..prefix.len()].copy_from_slice(prefix.as_bytes());
        s
    }

    println!("building large VarBinArray");
    let mut builder = VarBinBuilder::<i64>::with_capacity(N);
    let mut buf = vec![b'x'; STRING_LEN];
    for i in 0..N {
        let prefix = format!("row-{i:08}-");
        buf[..prefix.len()].copy_from_slice(prefix.as_bytes());
        builder.append_value(&buf);
    }
    let array = builder.finish(DType::Utf8(Nullability::NonNullable));

    println!("training FSST compressor");
    let compressor = fsst_train_compressor(&array);
    let len = array.len();
    let dtype = array.dtype().clone();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    println!("compressing to FSST");
    let compressed = fsst_compress(array, len, &dtype, &compressor, &mut ctx).into_array();

    println!("canonicalizing to VarBinView");
    let canonical = compressed.execute::<VarBinViewArray>(&mut ctx).unwrap();

    assert_eq!(
        canonical.len(),
        N,
        "row count must survive canonicalization"
    );
    assert!(
        canonical.data_buffers().len() >= 2,
        "decoded heap exceeding MAX_BUFFER_LEN must roll over into multiple buffers, got {}",
        canonical.data_buffers().len()
    );
    for (i, b) in canonical.data_buffers().iter().enumerate() {
        assert!(
            b.as_host().len() <= MAX_BUFFER_LEN,
            "buffer {i} of {} bytes exceeds MAX_BUFFER_LEN",
            b.as_host().len()
        );
    }

    // Spot-check the endpoints and the rows straddling the rollover boundary, which is the first
    // place the second buffer's offsets restart from zero.
    let boundary = MAX_BUFFER_LEN / STRING_LEN;
    for i in [0, boundary - 1, boundary, boundary + 1, N / 2, N - 1] {
        assert_eq!(
            canonical.bytes_at(i).as_slice(),
            nth_string(i).as_slice(),
            "value mismatch at row {i}"
        );
    }
}
