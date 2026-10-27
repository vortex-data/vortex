// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
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
    vortex_fsst::initialize(&session);
    session
});

// [(rows, minimum compressed KiB, taken rows)]. Every fixture has fewer than 1024 rows and takes
// one row or the most the sparse-take gate allows, so only the compressed-bytes gate chooses
// between gathering the taken rows and decoding every row.
const BENCH_ARGS: &[(usize, usize, usize)] = &[
    (16, 1, 1),
    (16, 8, 1),
    (16, 64, 1),
    (16, 1, 3),
    (16, 8, 3),
    (16, 64, 3),
    (256, 1, 1),
    (256, 8, 1),
    (256, 64, 1),
    (256, 1, 63),
    (256, 8, 63),
    (256, 64, 63),
    (1000, 1, 1),
    (1000, 8, 1),
    (1000, 64, 1),
    (1000, 1, 249),
    (1000, 8, 249),
    (1000, 64, 249),
];

const WORDS: &[&str] = &[
    "the", "a", "photo", "of", "with", "and", "on", "in", "small", "large", "red", "blue", "green",
    "white", "black", "dog", "cat", "man", "woman", "child", "table", "street", "car", "tree",
    "house", "sitting", "standing", "walking", "next", "to", "near", "under", "over", "wooden",
    "old", "new", "city", "park", "window", "light",
];

fn compressed(rows: usize, row_len: usize) -> FSSTArray {
    let mut rng = StdRng::seed_from_u64(0);
    let input = VarBinArray::from_iter(
        (0..rows).map(|_| {
            let mut row = String::with_capacity(row_len + 16);
            while row.len() < row_len {
                row.push_str(WORDS[rng.random_range(0..WORDS.len())]);
                row.push(' ');
            }
            row.truncate(row_len);
            Some(row)
        }),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = fsst_train_compressor(&input, &mut ctx).unwrap();
    fsst_compress(&input, &compressor, &mut ctx).unwrap()
}

fn fixture((rows, compressed_kib, taken): (usize, usize, usize)) -> (ArrayRef, ArrayRef) {
    // Grow the rows until their codes reach the target, so a fixture never lands just below a
    // threshold it is named after.
    let target = compressed_kib * 1024;
    let mut row_len = target / rows;
    let mut array = compressed(rows, row_len);
    while array.codes_bytes().len() < target {
        row_len += row_len.div_ceil(50);
        array = compressed(rows, row_len);
    }

    let stride = rows / taken;
    let indices = PrimitiveArray::from_iter((0..taken).map(|i| (i * stride + stride / 2) as u32));
    (array.into_array(), indices.into_array())
}

#[divan::bench(args = BENCH_ARGS)]
fn take(bencher: Bencher, args: (usize, usize, usize)) {
    let (array, indices) = fixture(args);
    bench_take(bencher, array, indices);
}

fn bench_take(bencher: Bencher, array: ArrayRef, indices: ArrayRef) {
    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        })
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    Short,
    AllLong,
    RepeatLong,
    HalfLong,
}

// (long rows among 256, long string length, selection). Short strings contain 8 bytes.
const UNEVEN_ARGS: &[(usize, usize, Selection)] = &[
    (1, 1024, Selection::Short),
    (1, 32768, Selection::Short),
    (1, 32768, Selection::AllLong),
    (16, 16384, Selection::AllLong),
    (16, 16384, Selection::RepeatLong),
    (16, 16384, Selection::HalfLong),
];

#[divan::bench(args = UNEVEN_ARGS)]
fn uneven_take(bencher: Bencher, args: (usize, usize, Selection)) {
    let (long_rows, long_len, mode) = args;
    let mut rng = StdRng::seed_from_u64(0);
    let input = VarBinArray::from_iter(
        (0..256).map(|i| {
            let len = if i < long_rows { long_len } else { 8 };
            let mut value = String::new();
            while value.len() < len {
                value.push_str(WORDS[rng.random_range(0..WORDS.len())]);
                value.push(' ');
            }
            value.truncate(len);
            Some(value)
        }),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = fsst_train_compressor(&input, &mut ctx).unwrap();
    let array = fsst_compress(&input, &compressor, &mut ctx).unwrap();
    let selected: Vec<u32> = match mode {
        Selection::Short => vec![long_rows as u32],
        Selection::AllLong => (0..long_rows as u32).collect(),
        Selection::RepeatLong => vec![0; 63],
        Selection::HalfLong => (0..(long_rows / 2) as u32).collect(),
    };
    bench_take(
        bencher,
        array.into_array(),
        PrimitiveArray::from_iter(selected).into_array(),
    );
}
