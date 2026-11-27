// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::hint::black_box;
use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use fastlanes::BitPacking;
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
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::patches::Patches;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_fastlanes::BitPacked;
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
    mark_optimized(&packed, &SESSION).unwrap();
    mark_optimized(&indices, &SESSION).unwrap();

    (packed, indices)
}

// Batch short iterations to reduce variation in the fastest walltime sample.
// The NEON cases still vary on unchanged code, so keep those measurements local.
// AVX2 with 80 selections still varies with batching. Retain its 64-selection case in CI.
#[vortex_bench_support::cpu_features(exclude = ["neon"])]
#[divan::bench(
    args = [64, 80].into_iter().filter(|&count| {
        env!("VORTEX_BENCH_VARIANT") != "avx2" || count != 80
    }),
    sample_size = 16,
)]
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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(&indices, &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(&indices, &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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
    mark_optimized(packed.as_array(), &SESSION).unwrap();
    mark_optimized(indices.as_array(), &SESSION).unwrap();

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

const HEAVY_PATCHED_LEN: usize = 64 * CHUNK_SIZE;
const HEAVY_PATCHED_BIT_WIDTH: u8 = 6;
const HEAVY_PATCHED_STRIDE: usize = 97;

fn heavy_patched_fixture(selectivity_pct: usize, sorted: bool) -> (ArrayRef, ArrayRef) {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..HEAVY_PATCHED_LEN).map(|index| {
        if index.is_multiple_of(HEAVY_PATCHED_STRIDE) {
            (500 + (index % 1_000)) as i16
        } else {
            (index % (1 << HEAVY_PATCHED_BIT_WIDTH)) as i16
        }
    }))
    .into_array();
    let packed = BitPackedData::encode(&values, HEAVY_PATCHED_BIT_WIDTH, &mut ctx).unwrap();
    assert_eq!(
        packed.patches().unwrap().num_patches(),
        HEAVY_PATCHED_LEN.div_ceil(HEAVY_PATCHED_STRIDE)
    );
    let packed = packed.into_array();

    let mut rng = StdRng::seed_from_u64(0);
    let count = HEAVY_PATCHED_LEN * selectivity_pct / 100;
    let range = Uniform::new(0, HEAVY_PATCHED_LEN as u32).unwrap();
    let mut raw_indices: Vec<u32> = (&mut rng).sample_iter(range).take(count).collect();
    if sorted {
        raw_indices.sort_unstable();
    }
    let indices = PrimitiveArray::from_iter(raw_indices).into_array();
    mark_optimized(&packed, &SESSION).unwrap();
    mark_optimized(&indices, &SESSION).unwrap();

    (packed, indices)
}

#[divan::bench(args = [1, 10])]
fn patched_take_heavy_sorted(bencher: Bencher, selectivity_pct: usize) {
    let (packed, indices) = heavy_patched_fixture(selectivity_pct, true);

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

#[divan::bench(args = [1, 10])]
fn patched_take_heavy_random(bencher: Bencher, selectivity_pct: usize) {
    let (packed, indices) = heavy_patched_fixture(selectivity_pct, false);

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

// These are the element runs emitted by a sparse FSL gather.
fn embedding_runs_fixture(dimension: usize, ptype: PType, bit_width: u8) -> (ArrayRef, ArrayRef) {
    const ROWS: usize = 512;
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
    bench_piecewise_take(bencher, packed, indices);
}

// The same runs appended to a builder through a lazy take, as an Arrow export of a fixed-size-list
// take does.
#[divan::bench(args = [1024])]
fn embedding_runs_append(bencher: Bencher, dimension: usize) {
    let (packed, indices) = embedding_runs_fixture(dimension, PType::U32, 16);
    mark_optimized(&packed, &SESSION).unwrap();
    mark_optimized(&indices, &SESSION).unwrap();

    bencher
        .counter(ItemsCount::new(indices.len()))
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            let lazy = packed.take(indices.clone()).unwrap();
            let mut builder = builder_with_capacity_in(lazy.dtype(), lazy.len(), ctx.allocator());
            lazy.append_to_builder(builder.as_mut(), ctx).unwrap();
            builder.finish()
        });
}

#[divan::bench(args = [512, 1024, 2048, 4096])]
fn embedding_code_runs(bencher: Bencher, dimension: usize) {
    let (packed, indices) = embedding_runs_fixture(dimension, PType::U8, 4);
    bench_piecewise_take(bencher, packed, indices);
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
    bench_piecewise_take(bencher, packed, indices);
}

// Isolated runs cannot amortize a whole-chunk decode across neighboring runs. Offsets near
// a chunk boundary also exercise sparse fragments of runs above the full-decode threshold.
#[divan::bench(
    types = [u8, u64],
    args = [
        (8, 0), (64, 0), (161, 0), (512, 0), (161, 944), (256, 1020),
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
    bench_piecewise_take(bencher, packed, indices);
}

fn bench_piecewise_take(bencher: Bencher, packed: ArrayRef, indices: ArrayRef) {
    mark_optimized(&packed, &SESSION).unwrap();
    mark_optimized(&indices, &SESSION).unwrap();

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

/// Numeric dictionary decoding with a 512-entry lookup table and 9-bit codes.
#[divan::bench(args = [false, true])]
fn dictionary_packed_codes(bencher: Bencher, fusion: bool) {
    let rows = 65_536u32;
    let code = |row: u32| (row.wrapping_mul(2_654_435_761) >> 23) as u16;
    let value = |index: u16| u64::from(index) * 12_001 + 781;
    let mut ctx = SESSION.create_execution_ctx();
    let codes = PrimitiveArray::from_iter((0..rows).map(code));
    let packed = BitPackedData::encode(&codes.into_array(), 9, &mut ctx).unwrap();
    let values = PrimitiveArray::from_iter((0..512).map(value));
    let dict = DictArray::try_new(packed.into_array(), values.into_array())
        .unwrap()
        .into_array();
    let result = dict.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert!(
        result
            .as_slice::<u64>()
            .iter()
            .copied()
            .eq((0..rows).map(code).map(value))
    );
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| {
            (
                dict.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fusion),
            )
        })
        .bench_values(|(dict, mut ctx)| dict.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// The repeated dictionary codes produced by RunEndArray::to_dict.
#[divan::bench(args = [false, true])]
fn dictionary_run_codes(bencher: Bencher, fusion: bool) {
    let runs = 4096u64;
    let lengths = (0..runs).map(|i| 1 + (i * 7) % 31);
    let rows = lengths.clone().sum::<u64>() as usize;
    let indices = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter(0..runs).into_array(),
        PrimitiveArray::from_iter(lengths.clone()).into_array(),
        ConstantArray::new(0u64, runs as usize).into_array(),
        rows,
    )
    .unwrap();
    let value = |i: u64| i.wrapping_mul(2_654_435_761);
    let dict = DictArray::try_new(
        indices.into_array(),
        PrimitiveArray::from_iter((0..runs).map(value)).into_array(),
    )
    .unwrap()
    .into_array();
    let result = dict
        .clone()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
        .unwrap();
    assert!(
        result.as_slice::<u64>().iter().copied().eq((0..runs)
            .zip(lengths)
            .flat_map(|(run, len)| std::iter::repeat_n(value(run), len as usize)))
    );
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| {
            (
                dict.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fusion),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// Control for plain dictionary decoding when no consumer requests a block source.
#[divan::bench(args = [false, true])]
fn dictionary_plain_codes(bencher: Bencher, fusion: bool) {
    let rows = 65_536u32;
    let codes =
        PrimitiveArray::from_iter((0..rows).map(|i| (i.wrapping_mul(2_654_435_761) >> 23) as u16));
    let table = PrimitiveArray::from_iter((0..512u64).map(|i| i * 12_001 + 781));
    let dict = DictArray::try_new(codes.into_array(), table.into_array())
        .unwrap()
        .into_array();
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| {
            (
                dict.clone(),
                SESSION.create_execution_ctx().with_block_fusion(fusion),
            )
        })
        .bench_values(|(dict, mut ctx)| dict.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// The two representations share the packed bytes and originate from the same exceptions.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PatchStorage {
    Native,
    Wrapper,
}

/// Sliced integer telemetry with one outlier per 211 rows.
#[divan::bench(args = [false, true])]
fn patched_packed_decode(bencher: Bencher, fusion: bool) {
    bench_patched_packed(bencher, PatchStorage::Wrapper, fusion, 65_536, 211);
}

#[divan::bench(args = [false, true])]
fn patched_packed_decode_million(bencher: Bencher, fusion: bool) {
    bench_patched_packed(bencher, PatchStorage::Wrapper, fusion, 1 << 20, 97);
}

#[divan::bench(args = [(262_144, false), (262_144, true), (524_288, false), (524_288, true)])]
fn patched_packed_window(bencher: Bencher, (rows, fusion): (u32, bool)) {
    bench_patched_packed(bencher, PatchStorage::Wrapper, fusion, rows, 97);
}

#[divan::bench(args = [(65_536, false), (65_536, true), (262_144, false), (262_144, true),
    (524_288, false), (524_288, true), (1_048_576, false), (1_048_576, true)])]
fn native_packed_decode(bencher: Bencher, (rows, fusion): (u32, bool)) {
    let stride = if rows == 65_536 { 211 } else { 97 };
    bench_patched_packed(bencher, PatchStorage::Native, fusion, rows, stride);
}

#[divan::bench(args = [false, true])]
fn wrapper_one_patch_per_chunk(bencher: Bencher, fusion: bool) {
    bench_patched_packed(bencher, PatchStorage::Wrapper, fusion, 1 << 20, 1024);
}

#[divan::bench]
fn native_one_patch_per_chunk(bencher: Bencher) {
    bench_patched_packed(bencher, PatchStorage::Native, false, 1 << 20, 1024);
}

fn bench_patched_packed(
    bencher: Bencher,
    storage: PatchStorage,
    fusion: bool,
    rows: u32,
    stride: usize,
) {
    let mut ctx = SESSION.create_execution_ctx();
    let reading = |row: u32| {
        if (row as usize).is_multiple_of(stride) {
            rows + row
        } else {
            row.wrapping_mul(2_654_435_761) >> 23
        }
    };
    let values = PrimitiveArray::from_iter((0..rows).map(reading));
    let native = BitPackedData::encode(&values.into_array(), 9, &mut ctx).unwrap();
    let patches = native.patches().unwrap();
    let packed = match storage {
        PatchStorage::Native => native.into_array(),
        PatchStorage::Wrapper => {
            let inner = BitPacked::try_new(
                native.packed().clone(),
                PType::U32,
                Validity::NonNullable,
                None,
                9,
                native.len(),
                native.offset(),
            )
            .unwrap();
            Patched::from_array_and_patches(inner.into_array(), &patches, &mut ctx)
                .unwrap()
                .into_array()
        }
    };
    let patched = packed.slice(31..rows as usize - 49).unwrap();
    let make_ctx = || SESSION.create_execution_ctx().with_block_fusion(fusion);
    #[cfg(not(codspeed))]
    {
        let traced = vortex_array::test_harness::trace::trace_op(|| {
            patched.clone().execute::<PrimitiveArray>(&mut make_ctx())
        })
        .unwrap();
        let trace = traced.trace.to_string();
        let direct = fusion && storage == PatchStorage::Wrapper;
        assert_eq!(
            trace.contains("fuse extend parent=vortex.patched"),
            direct,
            "{trace}"
        );
        assert_eq!(
            trace.contains("in_place u32->u32 output -> output"),
            direct,
            "{trace}"
        );
        assert!(
            traced
                .output
                .as_slice::<u32>()
                .iter()
                .copied()
                .eq((31..rows - 49).map(reading))
        );
    }
    bencher
        .counter(ItemsCount::new(patched.len()))
        .with_inputs(|| (patched.clone(), make_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// Full arrays with identical packed bytes. Clustered and spread both update 1,024 rows.
#[divan::bench(args = ["bare", "one_total_fused", "one_total_level", "spread_fused",
    "spread_level", "spread_native", "clustered_fused", "clustered_level"])]
fn sparse_patch_full_decode(bencher: Bencher, case: &str) {
    let rows = 1 << 20;
    let base = |row: usize| (row as u32).wrapping_mul(2_654_435_761) >> 23;
    let mut expected: Vec<u32> = (0..rows).map(base).collect();
    let mut ctx = SESSION.create_execution_ctx();
    let packed = BitPackedData::encode(
        &PrimitiveArray::from_iter(expected.iter().copied()).into_array(),
        9,
        &mut ctx,
    )
    .unwrap();
    let array = if case == "bare" {
        packed.into_array()
    } else {
        let mut rng = StdRng::seed_from_u64(491);
        let indices: Vec<u32> = if case.starts_with("one_total") {
            vec![(rows / 2) as u32]
        } else if case.starts_with("clustered") {
            (0..CHUNK_SIZE as u32)
                .map(|i| i + (rows / 2) as u32)
                .collect()
        } else {
            (0..rows / CHUNK_SIZE)
                .map(|i| (i * CHUNK_SIZE + rng.random_range(0..CHUNK_SIZE)) as u32)
                .collect()
        };
        let values: Vec<u32> = indices.iter().map(|&i| rows as u32 + i).collect();
        for (&index, &value) in indices.iter().zip(&values) {
            expected[index as usize] = value;
        }
        let patches = Patches::new(
            rows,
            0,
            PrimitiveArray::from_iter(indices).into_array(),
            PrimitiveArray::from_iter(values).into_array(),
            None,
            &mut ctx,
        )
        .unwrap();
        if case == "spread_native" {
            BitPacked::try_new(
                packed.packed().clone(),
                PType::U32,
                Validity::NonNullable,
                Some(patches),
                9,
                rows,
                0,
            )
            .unwrap()
            .into_array()
        } else {
            Patched::from_array_and_patches(packed.into_array(), &patches, &mut ctx)
                .unwrap()
                .into_array()
        }
    };
    let make_ctx = || {
        SESSION
            .create_execution_ctx()
            .with_block_fusion(case.ends_with("_fused"))
    };
    let decoded = array
        .clone()
        .execute::<PrimitiveArray>(&mut make_ctx())
        .unwrap();
    assert_eq!(decoded.as_slice::<u32>(), expected);
    drop(decoded);
    drop(expected);
    bencher
        .counter(ItemsCount::new(rows))
        .with_inputs(|| (array.clone(), make_ctx()))
        .bench_values(|(array, mut ctx)| array.execute::<PrimitiveArray>(&mut ctx).unwrap());
}

/// Isolate the 9-bit kernel and sparse stores on the same reusable 4 MiB output.
#[divan::bench(args = ["unpack_only", "patch_in_block", "patch_after_unpack"])]
fn sparse_patch_kernel(bencher: Bencher, case: &str) {
    const BLOCKS: usize = 1024;
    const PACKED_WORDS: usize = CHUNK_SIZE * 9 / u32::BITS as usize;
    let packed = vec![[0x92ab_715d_u32; PACKED_WORDS]; BLOCKS];
    let mut output = vec![[0u32; CHUNK_SIZE]; BLOCKS];
    let mut rng = StdRng::seed_from_u64(491);
    let indices: Vec<u16> = (0..BLOCKS)
        .map(|_| rng.random_range(0..CHUNK_SIZE) as u16)
        .collect();
    let values: Vec<u32> = (0..BLOCKS).map(|i| (1 << 20) + i as u32).collect();
    let unpack = |src: &[u32; PACKED_WORDS], dst: &mut [u32; CHUNK_SIZE]| {
        // SAFETY: src contains one 9-bit block and dst has exactly 1024 slots.
        unsafe { u32::unchecked_unpack(9, src, dst) };
    };
    bencher
        .counter(ItemsCount::new(BLOCKS * CHUNK_SIZE))
        .bench_local(|| {
            let output = black_box(&mut output[..]);
            match case {
                "patch_in_block" => {
                    for (((src, dst), &index), &value) in packed
                        .iter()
                        .zip(output.iter_mut())
                        .zip(&indices)
                        .zip(&values)
                    {
                        unpack(src, dst);
                        dst[usize::from(index)] = value;
                    }
                }
                "patch_after_unpack" | "unpack_only" => {
                    for (src, dst) in packed.iter().zip(output.iter_mut()) {
                        unpack(src, dst);
                    }
                    if case == "patch_after_unpack" {
                        for ((dst, &index), &value) in output.iter_mut().zip(&indices).zip(&values)
                        {
                            dst[usize::from(index)] = value;
                        }
                    }
                }
                _ => unreachable!(),
            }
            black_box(&*output);
        });
}
