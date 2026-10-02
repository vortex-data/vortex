// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode one BitPacked array with constant or per-block bit widths.
//!
//! Matching cases contain identical values. Constant packing uses the widest bit width in the
//! pattern; blocked packing uses primitive u32 offsets and a width per 1024-value block. Uniform
//! patterns therefore isolate offset-handling overhead with identical packed payloads.
//!
//! Inputs are non-nullable and unpatched. Construction and correctness checks are outside timing;
//! bulk decoding includes allocating and dropping the output. Cases are (length, offset, pattern).
//! The CPU-feature attribute measures the generated unpack kernels on each walltime CI leg.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench bitpacked_decode`.

#![expect(clippy::unwrap_used)]

use std::hint::black_box;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use fastlanes::BitPacking;
use mimalloc::MiMalloc;
use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::bitpack_compress::bitpack_primitive;
use vortex_fastlanes::bitpack_decompress::unpack_array;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Uniform7,
    Uniform20,
    Mixed,
    Zeros,
    Native,
}

impl Pattern {
    fn widths<T: NativePType>(self) -> [u8; 4] {
        match self {
            Self::Uniform7 => [7; 4],
            Self::Uniform20 => [20; 4],
            Self::Mixed => [3, 7, 12, 20],
            Self::Zeros => [0, 3, 0, 7],
            Self::Native => [0, 7, 16, u8::try_from(T::PTYPE.bit_width()).unwrap()],
        }
    }
}

const BULK_CASES: &[(usize, u16, Pattern)] = &[
    (1024, 0, Pattern::Uniform7),
    (1024, 0, Pattern::Uniform20),
    (65_536, 0, Pattern::Uniform7),
    (65_536, 0, Pattern::Uniform20),
    (1_048_576, 0, Pattern::Uniform7),
    (1_048_576, 0, Pattern::Uniform20),
    (65_536, 0, Pattern::Mixed),
    (1_048_576, 0, Pattern::Mixed),
    (65_500, 17, Pattern::Mixed),
    (65_536, 0, Pattern::Zeros),
    (65_536, 0, Pattern::Native),
];

const SCALAR_CASES: &[(usize, u16, Pattern)] = &[
    (1_048_576, 0, Pattern::Uniform7),
    (1_048_576, 0, Pattern::Mixed),
];

fn setup<T>(
    (len, offset, pattern): (usize, u16, Pattern),
    blocked: bool,
) -> (BitPackedArray, ExecutionCtx)
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    let widths = pattern.widths::<T>();
    let physical_len = len + usize::from(offset);
    let values: Vec<T> = (0..physical_len)
        .map(|i| {
            let width = widths[(i / FL_CHUNK_SIZE) % widths.len()];
            let mask = 1u64
                .checked_shl(u32::from(width))
                .map_or(u64::MAX, |bit| bit - 1);
            let value = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (value & mask).as_()
        })
        .collect();

    let array = if blocked {
        let mut packed = BufferMut::<T>::with_capacity(physical_len);
        let mut offsets = vec![128u32];
        for (block, values) in values.chunks(FL_CHUNK_SIZE).enumerate() {
            packed.extend_from_slice(&bitpack_primitive(values, widths[block % widths.len()]));
            offsets.push(128 + u32::try_from(packed.len() * size_of::<T>()).unwrap());
        }
        BitPacked::try_new_with_block_offsets(
            BufferHandle::new_host(packed.freeze().into_byte_buffer()),
            T::PTYPE,
            Validity::NonNullable,
            None,
            PrimitiveArray::from_iter(offsets).into_array(),
            len,
            offset,
        )
        .unwrap()
    } else {
        let bit_width = *widths.iter().max().unwrap();
        BitPacked::try_new(
            BufferHandle::new_host(bitpack_primitive(&values, bit_width).into_byte_buffer()),
            T::PTYPE,
            Validity::NonNullable,
            None,
            bit_width,
            len,
            offset,
        )
        .unwrap()
    };

    let mut ctx = SESSION.create_execution_ctx();
    let expected = &values[usize::from(offset)..];
    assert_eq!(
        unpack_array(array.as_view(), &mut ctx)
            .unwrap()
            .as_slice::<T>(),
        expected
    );
    for index in [0, len / 2, len - 1] {
        let expected_scalar: Scalar = expected[index].into();
        assert_eq!(
            array.execute_scalar(index, &mut ctx).unwrap(),
            expected_scalar
        );
    }
    (array, ctx)
}

fn bulk<T>(bencher: Bencher, case: (usize, u16, Pattern), blocked: bool)
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    let (array, mut ctx) = setup::<T>(case, blocked);
    bencher
        .counter(ItemsCount::new(array.len()))
        .bench_local(|| {
            black_box(unpack_array(black_box(array.as_view()), black_box(&mut ctx)).unwrap());
        });
}

fn scalar<T>(bencher: Bencher, case: (usize, u16, Pattern), blocked: bool)
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    let (array, mut ctx) = setup::<T>(case, blocked);
    let indices: Vec<usize> = (0..256).map(|i| (i * 104729 + 617) % array.len()).collect();
    bencher
        .counter(ItemsCount::new(indices.len()))
        .bench_local(|| {
            for &index in black_box(&indices) {
                black_box(
                    array
                        .execute_scalar(black_box(index), black_box(&mut ctx))
                        .unwrap(),
                );
            }
        });
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, u64], args = BULK_CASES)]
fn constant_width<T>(bencher: Bencher, case: (usize, u16, Pattern))
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    bulk::<T>(bencher, case, false);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, u64], args = BULK_CASES)]
fn blocked_widths<T>(bencher: Bencher, case: (usize, u16, Pattern))
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    bulk::<T>(bencher, case, true);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, u64], args = SCALAR_CASES)]
fn scalar_constant_width<T>(bencher: Bencher, case: (usize, u16, Pattern))
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    scalar::<T>(bencher, case, false);
}

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, u64], args = SCALAR_CASES)]
fn scalar_blocked_widths<T>(bencher: Bencher, case: (usize, u16, Pattern))
where
    T: NativePType + BitPacking + Into<Scalar>,
    u64: AsPrimitive<T>,
{
    scalar::<T>(bencher, case, true);
}
