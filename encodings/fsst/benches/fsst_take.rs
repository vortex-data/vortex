// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take from FSST values through a dictionary, the path a `Dict` over FSST values executes.

#![expect(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    vortex_fsst::initialize(&session);
    session
});

/// `(values_len, indices_len, index_range)`: indices are drawn uniformly from `0..index_range`.
const CASES: &[(&str, usize, usize, usize)] = &[
    // 1%, 2%, 5%, 10% and 20% of the rows, mostly distinct.
    ("sparse_1pct", 65_536, 655, 65_536),
    ("sparse_2pct", 65_536, 1_311, 65_536),
    ("sparse_5pct", 65_536, 3_277, 65_536),
    ("sparse_10pct", 65_536, 6_554, 65_536),
    ("sparse_20pct", 65_536, 13_107, 65_536),
    // Uniform over every row: about 21% and 37% of the indices repeat a value.
    ("random_50pct", 65_536, 32_768, 65_536),
    ("random_100pct", 65_536, 65_536, 65_536),
    // Every value referenced, each about 16 times.
    ("repeated_all", 4_096, 65_536, 4_096),
    // A quarter of the values referenced, each about 4 times.
    ("repeated_partial", 65_536, 65_536, 16_384),
    // Every value referenced once, in reverse.
    ("permutation", 65_536, 65_536, 0),
];

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 16
    }
}

fn values(len: usize) -> ArrayRef {
    let mut next = lcg(0x9e37_79b9_7f4a_7c15);
    let strings = (0..len)
        .map(|i| {
            format!(
                "https://www.example.com/products/{i:08x}?ref={:012x}",
                next()
            )
        })
        .collect::<Vec<_>>();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let mut ctx = SESSION.create_execution_ctx();
    let varbin = varbin.into_array();
    let compressor = fsst_train_compressor(&varbin, &mut ctx).unwrap();
    fsst_compress(&varbin, &compressor, &mut ctx)
        .unwrap()
        .into_array()
}

fn indices(values_len: usize, len: usize, range: usize) -> ArrayRef {
    if range == 0 {
        return PrimitiveArray::from_iter((0..values_len as u32).rev()).into_array();
    }
    let mut next = lcg(42);
    PrimitiveArray::from_iter((0..len).map(|_| (next() % range as u64) as u32)).into_array()
}

#[divan::bench(args = CASES.iter().map(|c| c.0))]
fn dict_execute(bencher: Bencher, case: &str) {
    let &(_, values_len, indices_len, range) = CASES.iter().find(|c| c.0 == case).unwrap();
    let dict = DictArray::try_new(indices(values_len, indices_len, range), values(values_len))
        .unwrap()
        .into_array();

    bencher
        .with_inputs(|| (dict.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(dict, mut ctx)| dict.execute::<VarBinViewArray>(&mut ctx).unwrap());
}
