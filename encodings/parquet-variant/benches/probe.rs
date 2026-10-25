// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use parquet_variant::VariantBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_parquet_variant::ParquetVariant;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const LEN: usize = 4096;
const ACCESSES: usize = 256;
const CASES: &[(&str, &str)] = &[
    ("typed", "random"),
    ("typed", "same_row"),
    ("raw", "random"),
    ("raw", "same_row"),
    ("mixed", "random"),
    ("mixed", "same_row"),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<KernelSession>();
    vortex_parquet_variant::initialize(&session);
    session
});

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn fixture(mode: &str) -> ArrayRef {
    let (raw_metadata, raw_value) = VariantBuilder::new().with_value("fallback").finish();
    let metadata = VarBinViewArray::from_iter_bin((0..LEN).map(|_| {
        if mode == "typed" {
            b"\x01\x00".as_slice()
        } else {
            raw_metadata.as_slice()
        }
    }))
    .into_array();
    let value = (mode != "typed").then(|| {
        VarBinViewArray::from_iter_nullable_bin((0..LEN).map(|index| {
            (mode == "raw" || index % 2 == 0).then_some(raw_value.as_slice())
        }))
        .into_array()
    });
    let typed_value = (mode != "raw").then(|| {
        PrimitiveArray::from_option_iter((0..LEN).map(|index| {
            (mode == "typed" || index % 2 == 1).then_some(index as i32)
        }))
        .into_array()
    });
    ParquetVariant::try_new(Validity::NonNullable, metadata, value, typed_value)
        .unwrap()
        .into_array()
}

fn read_index(pattern: &str, read: usize) -> usize {
    match pattern {
        "random" => (read * 7919) % LEN,
        "same_row" => 1000,
        _ => unreachable!(),
    }
}

#[divan::bench(args = CASES)]
fn one_off(bencher: Bencher, (mode, pattern): (&str, &str)) {
    let array = fixture(mode);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for read in 0..ACCESSES {
                divan::black_box(array.execute_scalar(read_index(pattern, read), ctx).unwrap());
            }
        });
}

#[divan::bench(args = CASES)]
fn repeated(bencher: Bencher, (mode, pattern): (&str, &str)) {
    let array = fixture(mode);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for read in 0..ACCESSES {
                divan::black_box(probe.execute_scalar(read_index(pattern, read), ctx).unwrap());
            }
        });
}
