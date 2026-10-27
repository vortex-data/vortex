// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray as _;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

const NUM_ARRAY_CHUNKS: usize = 64;
// Keep the selected count below the outer full-decode policy.
const NUM_SELECTED_CHUNKS: usize = 8;
const CHUNK_SIZE: usize = 1_024;
const THRESHOLD_FIXTURE_LEN: usize = NUM_ARRAY_CHUNKS * CHUNK_SIZE;

// Selection counts around the sparse extraction threshold for 16-bit values.
const THRESHOLD_BIT_WIDTH: u8 = 16;

fn threshold_fixture(selected_per_chunk: usize) -> (ArrayRef, ArrayRef) {
    let values: BufferMut<u32> = (0..THRESHOLD_FIXTURE_LEN)
        .map(|index| (index % (1 << THRESHOLD_BIT_WIDTH)) as u32)
        .collect();
    let packed = BitPackedData::encode(
        &PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array(),
        THRESHOLD_BIT_WIDTH,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap()
    .into_array();

    let indices = PrimitiveArray::from_iter((0..NUM_SELECTED_CHUNKS).flat_map(|chunk| {
        (0..selected_per_chunk)
            .map(move |index| (chunk * CHUNK_SIZE + index * CHUNK_SIZE / selected_per_chunk) as u32)
    }))
    .into_array();
    (packed, indices)
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = [8, 64, 80])]
fn threshold(bencher: Bencher, selected_per_chunk: usize) {
    let (packed, indices) = threshold_fixture(selected_per_chunk);

    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| (indices.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(indices, ctx)| {
            packed
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn take_10_stratified(bencher: Bencher) {
    let values = fixture(65_536, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter((0..10).map(|i| i * 6_553));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn take_10_contiguous(bencher: Bencher) {
    let values = fixture(65_536, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = buffer![0..10].into_array();

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn take_10k_random(bencher: Bencher) {
    let values = fixture(65_536, 8);
    let range = Uniform::new(0, values.len()).unwrap();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();

    let rng = StdRng::seed_from_u64(0);
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn take_10k_contiguous(bencher: Bencher) {
    let values = fixture(65_536, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter(0..10_000);

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn take_10k_dispersed(bencher: Bencher) {
    let values = fixture(65_536, 8);
    let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter((0..10_000).map(|i| (i * 42) % values.len() as u64));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn take_10k_first_chunk_only(bencher: Bencher) {
    let values = fixture(65_536, 8);
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter((0..10_000).map(|i| ((i * 42) % 1024) as u64));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

fn fixture(len: usize, bits: usize) -> Buffer<u32> {
    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0_u32, 2_u32.pow(bits as u32)).unwrap();
    rng.sample_iter(range).take(len).collect()
}

// There are currently 2 magic parameters of note:
// 1. the threshold at which sparse take will switch from search_sorted to map (currently 128)
// 2. the threshold at which bitpacked take will switch from bulk patching to per chunk patching (currently 64)
// There are thus 3 cases to consider:
// 1. N < 64 per chunk, covered by patched_take_10K_random
// 2. N > 128 per chunk, covered by patched_take_10K_contiguous_*
// 3. 64 < N < 128 per chunk, which is what we're trying to cover here (with 100 per chunk).
// As a result of the above, we get both search_sorted and per chunk patching, almost entirely on patches.
// I've iterated on both thresholds (1) and (2) using this collection of benchmarks, and those
// were roughly the best values that I found.

const BIG_BASE2: u32 = 65536;
const NUM_EXCEPTIONS: u32 = 1024;

#[divan::bench]
fn patched_take_10_stratified(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();

    assert!(packed.patches().is_some());
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        NUM_EXCEPTIONS as usize
    );

    let indices = PrimitiveArray::from_iter((0..10).map(|i| i * 6_653));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10_contiguous(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();

    assert!(packed.patches().is_some());
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        NUM_EXCEPTIONS as usize
    );

    let indices = buffer![0..10].into_array();

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10k_random(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();

    let rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0, values.len()).unwrap();
    let indices = PrimitiveArray::from_iter(rng.sample_iter(range).take(10_000).map(|i| i as u32));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10k_contiguous_not_patches(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter((0u32..NUM_EXCEPTIONS).cycle().take(10000));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10k_contiguous_patches(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();

    assert!(packed.patches().is_some());
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        NUM_EXCEPTIONS as usize
    );

    let indices =
        PrimitiveArray::from_iter((BIG_BASE2..BIG_BASE2 + NUM_EXCEPTIONS).cycle().take(10000));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10k_dispersed(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values.clone(), Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter((0..10_000).map(|i| (i * 42) % values.len() as u64));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10k_first_chunk_only(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let indices = PrimitiveArray::from_iter((0..10_000).map(|i| ((i * 42) % 1024) as u64));

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

#[divan::bench]
fn patched_take_10k_adversarial(bencher: Bencher) {
    let values = (0u32..BIG_BASE2 + NUM_EXCEPTIONS).collect::<Buffer<u32>>();
    let uncompressed = PrimitiveArray::new(values, Validity::NonNullable);
    let packed =
        bitpack_to_best_bit_width(&uncompressed, &mut SESSION.create_execution_ctx()).unwrap();
    let per_chunk_count = 100;
    let indices = PrimitiveArray::from_iter(
        (0..(NUM_EXCEPTIONS + 1024) / 1024)
            .cycle()
            .map(|chunk_idx| BIG_BASE2 - 1024 + chunk_idx * 1024)
            .flat_map(|base_idx| base_idx..(base_idx + per_chunk_count))
            .take(10000),
    );

    bencher
        .with_inputs(|| (&packed, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(packed, indices, execution_ctx)| {
            packed
                .take(indices.clone().into_array())
                .unwrap()
                .execute::<RecursiveCanonical>(execution_ctx)
                .unwrap()
        })
}

// These are the element runs emitted by a sparse FSL gather. The baseline materializes those
// runs inside the measured operation, reproducing the previous BitPacked take path.
fn embedding_runs_fixture(dimension: usize, ptype: PType, bit_width: u8) -> (ArrayRef, ArrayRef) {
    const ROWS: usize = 4096;
    const SELECTED: usize = 16;
    let mut ctx = SESSION.create_execution_ctx();
    let values = fixture(ROWS * dimension, usize::from(bit_width))
        .into_array()
        .cast(ptype.into())
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap()
        .into_array();
    let packed = BitPackedData::encode(&values, bit_width, &mut ctx)
        .unwrap()
        .into_array();
    let mut rng = StdRng::seed_from_u64(0);
    let starts = PrimitiveArray::from_iter(
        (0..SELECTED).map(|_| (rng.random_range(0..ROWS) * dimension) as u64),
    )
    .into_array();
    let indices = PiecewiseSequenceArray::try_new(
        starts,
        ConstantArray::new(dimension as u64, SELECTED).into_array(),
        ConstantArray::new(1u64, SELECTED).into_array(),
        SELECTED * dimension,
    )
    .unwrap()
    .into_array();
    (packed, indices)
}

#[divan::bench(args = [768, 1024, 1536])]
fn embedding_runs(bencher: Bencher, dimension: usize) {
    let (packed, indices) = embedding_runs_fixture(dimension, PType::U32, 16);
    bench_piecewise_take(bencher, packed, indices, false);
}

#[divan::bench(args = [768, 1024, 1536])]
fn embedding_runs_expanded_baseline(bencher: Bencher, dimension: usize) {
    let (packed, indices) = embedding_runs_fixture(dimension, PType::U32, 16);
    bench_piecewise_take(bencher, packed, indices, true);
}

#[divan::bench(args = [512, 1024, 2048, 4096])]
fn embedding_code_runs(bencher: Bencher, dimension: usize) {
    let (packed, indices) = embedding_runs_fixture(dimension, PType::U8, 4);
    bench_piecewise_take(bencher, packed, indices, false);
}

#[divan::bench(args = [512, 1024, 2048, 4096])]
fn embedding_code_runs_expanded_baseline(bencher: Bencher, dimension: usize) {
    let (packed, indices) = embedding_runs_fixture(dimension, PType::U8, 4);
    bench_piecewise_take(bencher, packed, indices, true);
}

// Short runs share chunks; long runs visit distant chunks. Neither crosses the full-decode
// threshold, including the largest case (16 * 1024 selected out of 128 * 1024 values).
fn fragmented_runs_fixture(run_length: usize) -> (ArrayRef, ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let values = fixture(128 * CHUNK_SIZE, 16).into_array();
    let packed = BitPackedData::encode(&values, 16, &mut ctx)
        .unwrap()
        .into_array();
    let (num_runs, stride) = if run_length < 64 {
        (128, run_length + 1)
    } else {
        (16, 4 * CHUNK_SIZE)
    };
    let starts =
        PrimitiveArray::from_iter((0..num_runs).map(|run| (run * stride) as u64)).into_array();
    let indices = PiecewiseSequenceArray::try_new(
        starts,
        ConstantArray::new(run_length as u64, num_runs).into_array(),
        ConstantArray::new(1u64, num_runs).into_array(),
        num_runs * run_length,
    )
    .unwrap()
    .into_array();
    (packed, indices)
}

#[divan::bench(args = [1, 2, 7, 8, 64, 512, 1024])]
fn fragmented_runs(bencher: Bencher, run_length: usize) {
    let (packed, indices) = fragmented_runs_fixture(run_length);
    bench_piecewise_take(bencher, packed, indices, false);
}

#[divan::bench(args = [1, 2, 7, 8, 64, 512, 1024])]
fn fragmented_runs_expanded_baseline(bencher: Bencher, run_length: usize) {
    let (packed, indices) = fragmented_runs_fixture(run_length);
    bench_piecewise_take(bencher, packed, indices, true);
}

// Isolated runs cannot amortize a whole-chunk decode across neighboring runs. Offsets near
// a chunk boundary also exercise sparse fragments of runs above the full-decode threshold.
#[divan::bench(
    types = [u8, u16, u32, u64],
    args = [
        (8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (160, 0), (161, 0),
        (256, 0), (512, 0), (161, 944), (256, 1020), (256, 896),
    ],
)]
fn isolated_runs<T: NativePType>(bencher: Bencher, (run_length, offset): (usize, usize)) {
    let bit_width = match T::PTYPE {
        PType::U8 => 4,
        PType::U16 => 9,
        PType::U32 => 16,
        PType::U64 => 33,
        _ => unreachable!(),
    };
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(42);
    let values = PrimitiveArray::from_iter(
        (0..128 * CHUNK_SIZE).map(|_| rng.random_range(0..(1u64 << bit_width))),
    )
    .into_array()
    .cast(T::PTYPE.into())
    .unwrap()
    .execute::<PrimitiveArray>(&mut ctx)
    .unwrap()
    .into_array();
    let packed = BitPackedData::encode(&values, bit_width, &mut ctx)
        .unwrap()
        .into_array();
    let num_runs = 16;
    let starts =
        PrimitiveArray::from_iter((0..num_runs).map(|run| (run * 4 * CHUNK_SIZE + offset) as u64))
            .into_array();
    let indices = PiecewiseSequenceArray::try_new(
        starts,
        ConstantArray::new(run_length as u64, num_runs).into_array(),
        ConstantArray::new(1u64, num_runs).into_array(),
        num_runs * run_length,
    )
    .unwrap()
    .into_array();
    bench_piecewise_take(bencher, packed, indices, false);
}

fn bench_piecewise_take(
    bencher: Bencher,
    packed: ArrayRef,
    indices: ArrayRef,
    expanded_baseline: bool,
) {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = packed.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_arrays_eq!(
        packed.take(indices.clone()).unwrap(),
        canonical.take(indices.clone()).unwrap(),
        &mut ctx
    );
    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| (indices.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(indices, ctx)| {
            let indices = if expanded_baseline {
                indices
                    .clone()
                    .execute::<PrimitiveArray>(ctx)
                    .unwrap()
                    .into_array()
            } else {
                indices.clone()
            };
            packed
                .take(indices)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
