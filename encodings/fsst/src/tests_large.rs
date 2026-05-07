// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stress regression tests for FSST compression at the i32-offset boundary.
//!
//! Gated to CI runs (collected but skipped when `CI` is unset; opt-out with
//! `VORTEX_SKIP_SLOW_TESTS=1`) because of the multi-GiB memory footprint.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;

use crate::fsst_compress;
use crate::fsst_train_compressor;

/// Regression for #7833: `fsst_compress` must accept inputs whose cumulative
/// compressed bytes exceed `i32::MAX`. Today this panics in
/// `vortex-array/src/arrays/varbin/builder.rs:62` because `fsst_compress_iter`
/// (`encodings/fsst/src/compress.rs:72`) hardcodes `VarBinBuilder::<i32>` for
/// the FSST output buffer regardless of input size.
///
/// The input is built with `VarBinBuilder::<i64>` to confirm that widening the
/// input alone does not help — the overflow is on the FSST output side.
///
/// `#[should_panic]` captures today's behavior; when the underlying bug is
/// fixed, drop the `#[should_panic]` so the trailing `assert_eq!` becomes the
/// regression assertion.
///
/// Allocates ~2.5 GiB for the input plus ~2.5 GiB for the FSST output.
#[test_with::env(CI)]
#[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
#[should_panic(expected = "to offset of type i32")]
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

    // Pre-fix: panics in `VarBinBuilder::<i32>::append_value` once cumulative
    // compressed bytes pass `i32::MAX`. Post-fix: must succeed with the row
    // count preserved.
    let compressed = fsst_compress(array, len, &dtype, &compressor, &mut ctx);
    assert_eq!(compressed.len(), len);
}
