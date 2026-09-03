// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic sweep comparing `BitPacked` (one bit width) against `BitPackedV2` (a width per
//! 1024-element chunk).
//!
//! Each case is 32 FastLanes chunks of one integer type, generated from a per-chunk width
//! pattern with optional exceptions and nulls. `compress_*` benches the width selection plus
//! packing, `decompress_*` the unpack back to a primitive array. Compressed sizes for every
//! case are printed to stderr before the timings.
//!
//! Run with: cargo bench -p vortex-fastlanes --bench bitpack_chunk_widths

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use num_traits::NumCast;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedV2Array;
use vortex_fastlanes::BitPackedV2ArrayExt;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_fastlanes::bitpacking_v2::bitpack_compress::bitpack_to_best_chunk_widths;
use vortex_session::VortexSession;

const NUM_CHUNKS: usize = 32;
const LEN: usize = NUM_CHUNKS * FL_CHUNK_SIZE;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

/// How the bit width needed by a chunk's values varies across the array.
#[derive(Clone, Copy)]
enum Pattern {
    /// Every chunk needs half the type width.
    Uniform,
    /// Widths grow linearly from 1 bit to nearly the full type width.
    Drift,
    /// Every chunk draws its own width at random.
    Random,
    /// Every other chunk is all zeros.
    ZeroHeavy,
    /// Narrow everywhere except one nearly full-width chunk in eight.
    Spiky,
}

impl Pattern {
    fn width(self, chunk: usize, bits: usize, rng: &mut StdRng) -> usize {
        match self {
            Pattern::Uniform => bits / 2,
            Pattern::Drift => 1 + chunk * (bits - 2) / NUM_CHUNKS,
            Pattern::Random => rng.random_range(1..bits),
            Pattern::ZeroHeavy => {
                if chunk.is_multiple_of(2) {
                    0
                } else {
                    bits / 2
                }
            }
            Pattern::Spiky => {
                if chunk % 8 == 7 {
                    bits - 1
                } else {
                    bits / 4
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    pattern: Pattern,
    /// Fraction of values pushed above their chunk's width.
    exceptions: f64,
    /// Fraction of null values.
    nulls: f64,
}

impl fmt::Debug for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.pattern {
            Pattern::Uniform => "uniform",
            Pattern::Drift => "drift",
            Pattern::Random => "random",
            Pattern::ZeroHeavy => "zero_heavy",
            Pattern::Spiky => "spiky",
        };
        write!(f, "{name}")?;
        if self.exceptions > 0.0 {
            write!(f, "+exc{}%", (self.exceptions * 100.0) as usize)?;
        }
        if self.nulls > 0.0 {
            write!(f, "+null{}%", (self.nulls * 100.0) as usize)?;
        }
        Ok(())
    }
}

const fn case(pattern: Pattern, exceptions: f64, nulls: f64) -> Case {
    Case {
        pattern,
        exceptions,
        nulls,
    }
}

const CASES: &[Case] = &[
    case(Pattern::Uniform, 0.0, 0.0),
    case(Pattern::Uniform, 0.01, 0.0),
    case(Pattern::Uniform, 0.0, 0.1),
    case(Pattern::Drift, 0.0, 0.0),
    case(Pattern::Drift, 0.01, 0.0),
    case(Pattern::Drift, 0.0, 0.1),
    case(Pattern::Random, 0.0, 0.0),
    case(Pattern::ZeroHeavy, 0.0, 0.0),
    case(Pattern::Spiky, 0.0, 0.0),
    case(Pattern::Spiky, 0.01, 0.0),
];

fn fixture<T: NativePType + NumCast>(case: Case) -> PrimitiveArray {
    let bits = T::PTYPE.bit_width();
    let mut rng = StdRng::seed_from_u64(42);
    let mut values = BufferMut::<T>::with_capacity(LEN);
    for chunk in 0..NUM_CHUNKS {
        let width = case.pattern.width(chunk, bits, &mut rng);
        for _ in 0..FL_CHUNK_SIZE {
            // An exception needs more bits than the chunk width but must still fit the type, so
            // a chunk already at the widest supported width cannot have any.
            let can_except = case.exceptions > 0.0 && width + 1 < bits;
            let v: u64 = if can_except && rng.random_bool(case.exceptions) {
                let exc_bits = (width + 8).min(bits - 1);
                rng.random_range((1u64 << width)..(1u64 << exc_bits))
            } else if width == 0 {
                0
            } else {
                rng.random_range(0..(1u64 << width))
            };
            values.push(T::from(v).unwrap());
        }
    }
    let validity = if case.nulls > 0.0 {
        Validity::from_iter((0..LEN).map(|_| !rng.random_bool(case.nulls)))
    } else {
        Validity::NonNullable
    };
    PrimitiveArray::new(values.freeze(), validity)
}

fn pack_v1(array: &PrimitiveArray) -> BitPackedArray {
    bitpack_to_best_bit_width(array, &mut SESSION.create_execution_ctx()).unwrap()
}

fn pack_v2(array: &PrimitiveArray) -> BitPackedV2Array {
    bitpack_to_best_chunk_widths(array, &mut SESSION.create_execution_ctx()).unwrap()
}

/// Compressed bytes of a v1 array: packed data and patches.
fn v1_bytes(array: &BitPackedArray) -> u64 {
    array.nbytes()
}

/// Compressed bytes of a v2 array: packed data, patches, and one width byte per chunk.
fn v2_bytes(array: &BitPackedV2Array) -> u64 {
    array.nbytes() + array.chunk_widths().len() as u64
}

fn v1_exceptions(array: &BitPackedArray) -> usize {
    array.patches().map_or(0, |p| p.num_patches())
}

fn v2_exceptions(array: &BitPackedV2Array) -> usize {
    array.patches().map_or(0, |p| p.num_patches())
}

fn report_sizes<T: NativePType + NumCast>() {
    for &case in CASES {
        let array = fixture::<T>(case);
        let v1 = pack_v1(&array);
        let v2 = pack_v2(&array);
        let (b1, b2) = (v1_bytes(&v1), v2_bytes(&v2));
        eprintln!(
            "{:<4} {:<22} raw {:>7}  v1 {:>7} B (bw {:>2}, {:>5} exc)  v2 {:>7} B (max {:>2}, {:>5} exc)  saving {:>6.2}%",
            T::PTYPE,
            format!("{case:?}"),
            array.nbytes(),
            b1,
            v1.bit_width(),
            v1_exceptions(&v1),
            b2,
            v2.bit_width(),
            v2_exceptions(&v2),
            100.0 * (b1 as f64 - b2 as f64) / b1 as f64,
        );
    }
}

fn main() {
    eprintln!("compressed sizes ({NUM_CHUNKS} chunks of {FL_CHUNK_SIZE} values per case):");
    report_sizes::<u8>();
    report_sizes::<u16>();
    report_sizes::<u32>();
    report_sizes::<u64>();
    divan::main();
}

#[divan::bench(types = [u8, u16, u32, u64], args = CASES)]
fn compress_v1<T: NativePType + NumCast>(bencher: Bencher, case: Case) {
    let array = fixture::<T>(case);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| bitpack_to_best_bit_width(array, &mut ctx).unwrap())
}

#[divan::bench(types = [u8, u16, u32, u64], args = CASES)]
fn compress_v2<T: NativePType + NumCast>(bencher: Bencher, case: Case) {
    let array = fixture::<T>(case);
    bencher
        .with_inputs(|| (&array, SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| bitpack_to_best_chunk_widths(array, &mut ctx).unwrap())
}

#[divan::bench(types = [u8, u16, u32, u64], args = CASES)]
fn decompress_v1<T: NativePType + NumCast>(bencher: Bencher, case: Case) {
    let packed = pack_v1(&fixture::<T>(case)).into_array();
    bencher
        .with_inputs(|| (packed.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(packed, mut ctx)| packed.execute::<PrimitiveArray>(&mut ctx).unwrap())
}

#[divan::bench(types = [u8, u16, u32, u64], args = CASES)]
fn decompress_v2<T: NativePType + NumCast>(bencher: Bencher, case: Case) {
    let packed = pack_v2(&fixture::<T>(case)).into_array();
    bencher
        .with_inputs(|| (packed.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(packed, mut ctx)| packed.execute::<PrimitiveArray>(&mut ctx).unwrap())
}
