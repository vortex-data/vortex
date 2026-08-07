// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalizing `DecimalByteParts` into `i128`/`i256` values.
//!
//! Reassembly walks a most significant part plus one (`i128`) or three (`i256`) unsigned
//! 64-bit lower parts and produces one wide value per row. This benchmark measures the
//! shipped path through the array API, so it tracks whatever shape the crate currently uses
//! and cannot drift away from it.
//!
//! Alternative shapes were compared while choosing that path and then removed, since keeping
//! hand-written copies of the assembly loop here means maintaining the same loop twice. At
//! 65,536 rows, `fastest` of three runs:
//!
//! - **`i256` is dominated by the part count being visible to the compiler.** Specializing it
//!   to a constant is 1.85x (190 µs against 351 µs). How the output is written barely matters
//!   at 32 bytes per row.
//! - **`i128` is dominated by the write.** Specializing the part count is worth only ~1.04x,
//!   while storing into a pre-sized buffer instead of pushing into a reserved one is 1.6x
//!   (83 µs against 138 µs) — the bounds-checked `push` is the whole cost at 16 bytes per row.
//! - **Columnar always loses.** For `i256` each lane store is strided by 32 bytes, 2.3x slower
//!   than the row loop (438 µs); cache blocking the passes recovered part of that and was
//!   still 1.6x slower; expressing them as whole-value `i256` shifts was 11x slower. For
//!   `i128` the two-pass column shape (103 µs) beats the *pushing* row loop but still loses to
//!   the single-pass write, so the second pass buys nothing once the push is gone.
//! - **Hand-written 64-bit words do not beat the `u128` packing.** `i256::from_parts` takes a
//!   `u128` and an `i128`, so each row ends in `u128::from(w0) | (u128::from(w1) << 64)`.
//!   Writing four `u64` lanes by hand instead ties it. Disassembly says why: neither emits a
//!   single `shld`/`shrd`, and both compile to four plain 64-bit stores per row at offsets
//!   0x0/0x8/0x10/0x18. The `i128` loop is the same — `(i128::from(msp) << 64) | i128::from(p)`
//!   becomes two 64-bit stores. A shift by a constant multiple of 64 followed by an or is pure
//!   data movement and LLVM recognizes it; the 128-bit codegen worth avoiding is division and
//!   remainder, which call into compiler-rt, and shifts by a runtime amount. Neither is here.
//!
//! The removed variants are recoverable from git history if a future change needs to re-run
//! the comparison rather than trust these numbers.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_decimal_byte_parts::DecimalByteParts;

fn main() {
    divan::main();
}

/// Rows per benchmark: a typical scan chunk, and large enough that the output does not fit
/// in L2.
const LEN: usize = 65_536;

/// Deterministic pseudo-random words, so no part is constant or a sequence.
fn words(seed: u64, len: usize) -> Buffer<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random()).collect()
}

fn msp(seed: u64, len: usize) -> Buffer<i64> {
    words(seed, len)
        .iter()
        .map(|w| (w >> 40).cast_signed())
        .collect()
}

/// Canonicalizing through the public array API, so the child execution and validity handling
/// around the assembly loop are included.
#[divan::bench(args = [1, 3])]
fn canonicalize_byte_parts(bencher: Bencher, lower_parts: usize) {
    let msp = PrimitiveArray::new(msp(1, LEN), Validity::NonNullable);
    let lower = (0..lower_parts)
        .map(|i| PrimitiveArray::new(words(7 + i as u64, LEN), Validity::NonNullable))
        .collect::<Vec<_>>();

    let dtype = if lower_parts == 1 {
        DecimalDType::new(38, 2)
    } else {
        DecimalDType::new(76, 2)
    };
    let array = DecimalByteParts::try_new_with_lower_parts(
        msp.into_array(),
        lower.into_iter().map(IntoArray::into_array).collect(),
        dtype,
    )
    .unwrap()
    .into_array();

    let session = array_session();
    bencher
        .with_inputs(|| session.create_execution_ctx())
        .bench_refs(|ctx| {
            black_box(array.clone())
                .execute::<DecimalArray>(ctx)
                .unwrap()
        });
}
