// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sweeps the public `BitPackedArray` compare-against-constant path (`array.binary(rhs, op)`) over
//! every integer type and every valid bit width, so a kernel change shows up as a CodSpeed diff.
//!
//! The array holds in-range values (no patches, no out-of-range fast path), so each iteration runs
//! the full unpack + per-element compare kernel that backs `<BitPacked as CompareKernel>`.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpack_compare_sweep`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Number of elements per benchmarked array (64 full FastLanes blocks).
const LEN: usize = 64 * 1024;

/// Operator under test. `Lt` exercises the full unpack + per-element comparison path.
const OP: Operator = Operator::Lt;

/// Integer types we can build packed arrays for in the benchmark.
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

impl_bench_int!(u8, u16, u32, u64, i8, i16, i32, i64);

/// Rebuild the array with its packed buffer copied to a page-aligned allocation.
///
/// `bitpack_encode` aligns the packed buffer only to the element type, so its cache-line
/// placement depends on allocator state, which shifts with any change to the bench binary.
/// CodSpeed's simulated cache misses are deterministic in those addresses, which made the whole
/// sweep flip ~40% between two layout modes across unrelated commits. Pinning the buffer to a
/// page boundary makes the layout, and therefore the measurement, reproducible.
fn page_aligned(array: BitPackedArray) -> BitPackedArray {
    let ptype = array.dtype().as_ptype();
    let parts = BitPacked::into_parts(array);
    BitPacked::try_new(
        parts.packed.ensure_aligned(Alignment::new(4096)).unwrap(),
        ptype,
        parts.validity,
        parts.patches,
        parts.bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

/// Encode `LEN` in-range values of type `T` at the given bit width, returning the packed array, a
/// mid-range constant to compare against, and an execution context.
fn setup<T: BenchInt>(width: usize) -> (ArrayRef, ArrayRef, ExecutionCtx) {
    let mut ctx = SESSION.create_execution_ctx();
    let cap = 1u64 << width;
    let buf: BufferMut<T> = (0..LEN)
        .map(|i| T::from_counter((i as u64) % cap))
        .collect();
    let array = page_aligned(
        BitPackedData::encode(
            &PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array(),
            width as u8,
            &mut ctx,
        )
        .unwrap(),
    )
    .into_array();
    let rhs = ConstantArray::new(T::from_counter(cap / 2), LEN).into_array();
    (array, rhs, ctx)
}

/// Generate a compare benchmark over every valid bit width for one type. Valid widths are
/// `1..native_bits` - bit-packing requires the target width to be strictly narrower than the type.
macro_rules! bench_type {
    ($modname:ident, $T:ty, $native_bits:expr) => {
        mod $modname {
            use super::*;

            #[divan::bench(args = 1..$native_bits)]
            fn compare(bencher: Bencher, width: usize) {
                let (array, rhs, mut ctx) = setup::<$T>(width);
                bencher.counter(ItemsCount::new(LEN)).bench_local(|| {
                    array
                        .clone()
                        .binary(rhs.clone(), OP)
                        .unwrap()
                        .execute::<BoolArray>(&mut ctx)
                        .unwrap()
                });
            }
        }
    };
}

bench_type!(u8, u8, 8usize);
bench_type!(u16, u16, 16usize);
bench_type!(u32, u32, 32usize);
bench_type!(u64, u64, 64usize);
bench_type!(i8, i8, 8usize);
bench_type!(i16, i16, 16usize);
bench_type!(i32, i32, 32usize);
bench_type!(i64, i64, 64usize);
