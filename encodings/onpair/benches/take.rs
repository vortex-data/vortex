// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take from OnPair values through a dictionary, the path a `Dict` over OnPair values executes.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_onpair::initialize(&session);
    session
});

const ROWS: u32 = 2048;
const HOT_ROWS: u32 = ROWS / 8;

static VALUES: LazyLock<ArrayRef> = LazyLock::new(|| {
    let strings = VarBinArray::from_iter(
        (0..ROWS).map(|row| {
            let reference = row.wrapping_mul(0x9e37_79b1);
            Some(format!(
                "https://example.com/products/{row:08x}?ref={reference:08x}"
            ))
        }),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    onpair_compress(&strings, DEFAULT_CONFIG, &mut ctx).unwrap()
});

fn draws() -> impl Iterator<Item = u32> {
    let mut state = 42u32;
    (0..ROWS).map(move |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state >> 12
    })
}

#[divan::bench]
fn repeated_scattered(bencher: Bencher) {
    let codes = PrimitiveArray::from_iter(draws().map(|draw| (draw % HOT_ROWS) * 137 % ROWS));
    bench_dict(bencher, codes);
}

#[divan::bench]
fn nullable_prefix(bencher: Bencher) {
    let codes = PrimitiveArray::from_option_iter(
        draws().map(|draw| (draw & HOT_ROWS == 0).then_some(draw % HOT_ROWS)),
    );
    bench_dict(bencher, codes);
}

fn bench_dict(bencher: Bencher, codes: PrimitiveArray) {
    let dict = DictArray::new(codes.into_array(), VALUES.clone()).into_array();
    bencher
        .with_inputs(|| (dict.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(dict, mut ctx)| dict.execute::<VarBinViewArray>(&mut ctx).unwrap());
}
