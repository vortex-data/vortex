// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalize a dictionary whose codes are null-filled sparse patches.

#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

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
    vortex_sparse::initialize(&session);
    session
});

const LEN: usize = 65_536;
const DICT_SIZE: usize = 1_000;

/// Distance between present rows: every 100th row (1%) or every 10th row (10%).
const PATCH_STRIDES: &[usize] = &[100, 10];

/// `Dict(Sparse(positions, codes, NULL), values)` with one present row every `patch_stride` rows.
///
/// The array is left unoptimized, as a reader produces it, so canonicalization chooses how to
/// combine the dictionary with its sparse codes.
fn make_dict_of_sparse_codes(patch_stride: usize, values: ArrayRef) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let positions: Buffer<u32> = (0..LEN).step_by(patch_stride).map(|i| i as u32).collect();
    // Spread the present codes over the dictionary, with repeats.
    let codes =
        PrimitiveArray::from_iter((0..positions.len()).map(|i| ((i * 7_919) % DICT_SIZE) as u16))
            .into_array();
    let sparse_codes = Sparse::try_new(
        positions.into_array(),
        codes,
        LEN,
        Scalar::null_native::<u16>(),
        &mut ctx,
    )
    .vortex_expect("sparse codes should be valid")
    .into_array();

    DictArray::try_new(sparse_codes, values)
        .vortex_expect("dictionary over sparse codes should be valid")
        .into_array()
}

#[divan::bench(args = PATCH_STRIDES)]
fn canonicalize_dict_sparse_codes_primitive(bencher: Bencher, patch_stride: usize) {
    let values = PrimitiveArray::from_iter((0i64..).step_by(1_000).take(DICT_SIZE)).into_array();
    bench_canonicalize(bencher, make_dict_of_sparse_codes(patch_stride, values));
}

#[divan::bench(args = PATCH_STRIDES)]
fn canonicalize_dict_sparse_codes_utf8(bencher: Bencher, patch_stride: usize) {
    let values = VarBinViewArray::from_iter_str(
        (0..DICT_SIZE).map(|i| format!("dictionary value number {i:04}")),
    )
    .into_array();
    bench_canonicalize(bencher, make_dict_of_sparse_codes(patch_stride, values));
}

fn bench_canonicalize(bencher: Bencher, dict: ArrayRef) {
    bencher
        .with_inputs(|| (dict.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(
                array
                    .execute::<Canonical>(&mut ctx)
                    .vortex_expect("dictionary over sparse codes canonicalization"),
            )
        });
}
