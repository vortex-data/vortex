// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

// Sized to keep CodSpeed simulation under 1ms per benchmark.
const LEN: usize = 2_048;

/// Benchmarks fill_null on a UTF-8 array with a fill value short enough to inline in its view.
#[divan::bench]
fn varbinview_fill_null_inlined(bencher: Bencher) {
    bench_fill_null(bencher, "fill");
}

/// Benchmarks fill_null on a UTF-8 array with a fill value too long to inline in its view.
#[divan::bench]
fn varbinview_fill_null_outlined(bencher: Bencher) {
    bench_fill_null(bencher, "a fill value that is too long to inline");
}

fn bench_fill_null(bencher: Bencher, fill: &str) {
    let array = fixture(LEN);
    let fill = Scalar::utf8(fill, Nullability::NonNullable);

    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .fill_null(fill.clone())
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

fn fixture(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter(
        [
            Some("short"),
            None,
            Some("this is a much longer string to force outlining"),
        ]
        .into_iter()
        .cycle()
        .take(len),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array()
}
