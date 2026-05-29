// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::CompressorBuilder;
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

// TODO(someone): ideally CI would run this in release mode as well since debug builds make the
// allocation and compression loop substantially slower.
/// Regression for #7833: [`fsst_compress`] must accept inputs whose cumulative compressed
/// bytes exceed [`i32::MAX`]. Before the fix, [`fsst_compress_iter`] hardcoded
/// [`VarBinBuilder<i32>`] for the FSST output and panicked in
/// [`VarBinBuilder::append_value`] once cumulative compressed bytes crossed the boundary.
///
/// We force the output past [`i32::MAX`] with an empty FSST compressor: it has no symbols, so
/// every input byte is emitted as a two-byte escape and the compressed output is
/// deterministically `2 *` the input size. This crosses the boundary with ~1.1 GiB of input
/// rather than the ~2.5 GiB of incompressible data a trained compressor would require, which
/// roughly halves the compression work and removes random-data generation entirely while
/// exercising the same output-offset path. The escape factor is the worst case FSST can
/// produce (`2 * len + 7`), so this is also the cheapest way to reach the boundary.
///
/// The input offsets stay below [`i32::MAX`], so the input build never overflows; only the FSST
/// output crosses the boundary, isolating the regression to the output side. The test asserts
/// the actual compressed byte size exceeds [`i32::MAX`] so it cannot silently stop covering the
/// regression if FSST's escape behavior ever changes.
///
/// Allocates ~1.1 GiB for the input and ~2.25 GiB for the FSST output (~3.4 GiB total), so it
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
    const STRING_LEN: usize = 64 * 1024;
    // An empty compressor escapes every byte, so each 64 KiB string compresses to exactly
    // 128 KiB of output. Target ~2.25 GiB of output (2^31 + 256 MiB margin) so cumulative
    // compressed bytes comfortably exceed i32::MAX.
    const TOTAL_OUTPUT_BYTES: usize = (1usize << 31) + (256 << 20);
    const N: usize = TOTAL_OUTPUT_BYTES / (2 * STRING_LEN);

    // The content is irrelevant because the empty compressor escapes every byte; `b'a'` keeps
    // the input valid UTF-8. A single reused buffer avoids per-row allocation.
    let string = vec![b'a'; STRING_LEN];

    println!("building large VarBinArray");
    let mut builder = VarBinBuilder::<i64>::with_capacity(N);
    for _ in 0..N {
        builder.append_value(&string);
    }
    let array = builder.finish(DType::Utf8(Nullability::NonNullable));

    // Empty symbol table -> every byte is escaped -> 2x expansion.
    let compressor = CompressorBuilder::default().build();
    let len = array.len();
    let dtype = array.dtype().clone();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    println!("compressing to FSST");
    let compressed = fsst_compress(array, len, &dtype, &compressor, &mut ctx);
    assert_eq!(compressed.len(), len);

    // The regression is only exercised if cumulative compressed bytes truly exceed i32::MAX.
    let compressed_bytes = compressed.codes_bytes().len();
    assert!(
        compressed_bytes > i32::MAX as usize,
        "compressed output ({compressed_bytes} bytes) must exceed i32::MAX to require i64 offsets",
    );
}
