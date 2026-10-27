// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression and decompression of deterministic synthetic numeric arrays.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySessionExt;
use vortex_error::VortexExpect;
use vortex_pco::Pco;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LENGTH: usize = 65_536;
const VALUES_PER_PAGE: usize = 8192;
const COLUMNS: [Column; 4] = [
    Column::NullablePeriodic,
    Column::SmallIntegers,
    Column::WideIntegers,
    Column::RandomDoubles,
];

#[derive(Clone, Copy, Debug)]
enum Column {
    NullablePeriodic,
    SmallIntegers,
    WideIntegers,
    RandomDoubles,
}

struct Fixture {
    input: PrimitiveArray,
    compressed: ArrayRef,
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>();
    session.arrays().register(Pco);
    session
});

fn input(column: Column) -> PrimitiveArray {
    let mut state = 42u64;
    let random = (0..LENGTH).map(move |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state
    });
    match column {
        Column::NullablePeriodic => PrimitiveArray::from_option_iter(
            (0..LENGTH).map(|i| (i % 13 != 0).then_some((i % 101) as i32)),
        ),
        Column::SmallIntegers => {
            PrimitiveArray::from_iter(random.map(|value| ((value >> 32) % 101) as i32))
        }
        Column::WideIntegers => {
            PrimitiveArray::from_iter(random.map(|value| (value >> 32) as i32))
        }
        Column::RandomDoubles => PrimitiveArray::from_iter(
            random.map(|value| (value >> 11) as f64 / (1u64 << 53) as f64),
        ),
    }
}

static FIXTURES: LazyLock<[Fixture; 4]> = LazyLock::new(|| {
    COLUMNS.map(|column| {
        let input = input(column);
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = Pco::from_primitive(
            input.as_view(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            VALUES_PER_PAGE,
            &mut ctx,
        )
        .vortex_expect("PCO fixture compression")
        .into_array();
        let decoded = compressed
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .vortex_expect("PCO fixture decompression");
        assert_arrays_eq!(decoded, input.clone(), &mut ctx);
        Fixture { input, compressed }
    })
});

fn fixture(column: Column) -> &'static Fixture {
    &FIXTURES[column as usize]
}

fn main() {
    LazyLock::force(&FIXTURES);
    divan::main();
}

#[divan::bench(args = COLUMNS)]
fn compress(bencher: Bencher, column: Column) {
    let input = &fixture(column).input;
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_values(|mut ctx| {
            Pco::from_primitive(
                input.as_view(),
                pco::DEFAULT_COMPRESSION_LEVEL,
                VALUES_PER_PAGE,
                &mut ctx,
            )
            .vortex_expect("PCO compression")
        });
}

#[divan::bench(args = COLUMNS)]
fn decompress(bencher: Bencher, column: Column) {
    let compressed = &fixture(column).compressed;
    bencher
        .with_inputs(|| (compressed.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("PCO decompression")
        });
}
