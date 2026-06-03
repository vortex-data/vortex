// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sweeps the public `BitPackedArray` compare-against-constant path (`array.binary(rhs, op)`) over
//! every unsigned integer type and a representative set of bit widths, so a kernel change shows up
//! as a CodSpeed diff.
//!
//! The array holds in-range values (no patches, no out-of-range fast path), so each iteration runs
//! the full unpack + per-element compare kernel that backs `<BitPacked as CompareKernel>`.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpack_compare_sweep`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedData;

fn main() {
    divan::main();
}

/// Number of elements per benchmarked array (64 full FastLanes blocks).
const LEN: usize = 64 * 1024;

/// Operator under test. `Lt` exercises the full unpack + per-element comparison path.
const OP: Operator = Operator::Lt;

/// Bit widths to sweep. A spread of small widths plus a few wide ones; widths that do not fit a
/// given type are filtered out per type (bit-packing requires `width < T::BITS`).
const WIDTHS: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 15, 31, 40, 62];

/// Widths from [`WIDTHS`] that can actually be packed into a `native_bits`-wide type.
fn valid_widths(native_bits: u32) -> Vec<u8> {
    WIDTHS
        .iter()
        .copied()
        .filter(|&w| u32::from(w) < native_bits)
        .collect()
}

/// Unsigned integer types we can build packed arrays for in the benchmark.
trait BenchInt: NativePType + Copy + Into<Scalar> {
    /// Build an in-range value from a small counter.
    fn from_counter(v: u64) -> Self;
}

macro_rules! impl_bench_int {
    ($($T:ty),+) => {
        $(impl BenchInt for $T {
            #[inline]
            fn from_counter(v: u64) -> Self {
                v as $T
            }
        })+
    };
}

impl_bench_int!(u8, u16, u32, u64);

/// Encode `LEN` in-range values of type `T` at the given bit width, returning the packed array, a
/// mid-range constant to compare against, and an execution context.
fn setup<T: BenchInt>(width: u8) -> (ArrayRef, ArrayRef, ExecutionCtx) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let cap = 1u64 << width;
    let buf: BufferMut<T> = (0..LEN)
        .map(|i| T::from_counter((i as u64) % cap))
        .collect();
    let array = BitPackedData::encode(
        &PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array(),
        width,
        &mut ctx,
    )
    .unwrap()
    .into_array();
    let rhs = ConstantArray::new(T::from_counter(cap / 2), LEN).into_array();
    (array, rhs, ctx)
}

/// Shared benchmark body: build the inputs outside the timed region (`with_inputs`) and time the
/// compare kernel over them by reference (`bench_local_refs`).
fn run<T: BenchInt>(bencher: Bencher, width: u8) {
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| setup::<T>(width))
        .bench_local_refs(|(array, rhs, ctx)| {
            array
                .clone()
                .binary(rhs.clone(), OP)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

/// Emit one divan benchmark per unsigned type, sweeping only the widths that fit the type.
macro_rules! bench_unsigned {
    ($($T:ident),+ $(,)?) => {
        $(
            #[divan::bench(args = valid_widths($T::BITS))]
            fn $T(bencher: Bencher, width: u8) {
                run::<$T>(bencher, width);
            }
        )+
    };
}

bench_unsigned!(u8, u16, u32, u64);
