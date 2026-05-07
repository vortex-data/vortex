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

/// Regression for #7833: `fsst_compress` must accept inputs whose cumulative
/// compressed bytes exceed `i32::MAX`. Pre-fix, `fsst_compress_iter` hardcoded
/// `VarBinBuilder::<i32>` for the FSST output buffer regardless of input size,
/// which panicked in `VarBinBuilder::<i32>::append_value` once cumulative
/// compressed bytes passed `i32::MAX`.
///
/// Allocates ~2.5 GiB for the input plus ~2.5 GiB for the FSST output, so the
/// test is `#[ignore]`-d by default. Run explicitly with:
/// `cargo test --release -p vortex-fsst -- --ignored fsst_compress_offsets`.
#[test]
#[ignore = "allocates ~5 GiB; run with --ignored"]
fn fsst_compress_offsets_overflow_i32() {
    // High-entropy ASCII strings sliced from a random pool. FSST is a
    // symbol-table compressor; pseudo-random data with no recurring byte
    // sequences resists compression, so the compressed output stays close
    // to input size and crosses the i32 boundary.
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

    let mut builder = VarBinBuilder::<i64>::with_capacity(N);
    for i in 0..N {
        let off = (i.wrapping_mul(31337)) % (POOL_LEN - STRING_LEN);
        builder.append_value(&pool[off..off + STRING_LEN]);
    }
    let array = builder.finish(DType::Utf8(Nullability::NonNullable));

    let compressor = fsst_train_compressor(&array);
    let len = array.len();
    let dtype = array.dtype().clone();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    let compressed = fsst_compress(array, len, &dtype, &compressor, &mut ctx);
    assert_eq!(compressed.len(), len);
}
