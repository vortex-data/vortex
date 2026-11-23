// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonicalize a dictionary whose codes are null-filled sparse patches.

use std::iter::successors;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
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
        .with::<KernelSession>();
    vortex_sparse::initialize(&session);
    session
});

const LEN: u32 = 8_192;
const DICT_SIZE: u16 = 1_000;
/// Step between consecutive present codes. It is coprime with `DICT_SIZE`, so the codes cycle
/// through the whole dictionary.
const CODE_STEP: u16 = 919;

/// Distance between present rows: every 100th row (1%) or every 10th row (10%).
const PATCH_STRIDES: &[usize] = &[100, 10];

/// `Dict(Sparse(positions, codes, NULL), values)` with one present row every `patch_stride` rows.
///
/// The array is left unoptimized, as a reader produces it, so canonicalization chooses how to
/// combine the dictionary with its sparse codes.
fn make_dict_of_sparse_codes(patch_stride: usize, values: &ArrayRef) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let positions: Buffer<u32> = (0..LEN).step_by(patch_stride).collect();
    // Spread the present codes over the dictionary, with repeats. The codes share the null
    // fill's nullable dtype.
    let codes = PrimitiveArray::from_option_iter(
        successors(Some(0u16), |code| Some((code + CODE_STEP) % DICT_SIZE))
            .take(positions.len())
            .map(Some),
    )
    .into_array();
    let sparse_codes = Sparse::try_new(
        positions.into_array(),
        codes,
        LEN as usize,
        Scalar::null_native::<u16>(),
        &mut ctx,
    )
    .vortex_expect("sparse codes should be valid")
    .into_array();

    DictArray::try_new(sparse_codes, values.clone())
        .vortex_expect("dictionary over sparse codes should be valid")
        .into_array()
}

static PRIMITIVE_VALUES: LazyLock<ArrayRef> = LazyLock::new(|| {
    PrimitiveArray::from_iter((0..i64::from(DICT_SIZE)).map(|i| i * 1_000)).into_array()
});

static UTF8_VALUES: LazyLock<ArrayRef> = LazyLock::new(|| {
    VarBinViewArray::from_iter_str(
        (0..DICT_SIZE).map(|i| format!("dictionary value number {i:04}")),
    )
    .into_array()
});

#[divan::bench(args = PATCH_STRIDES)]
fn canonicalize_dict_sparse_codes_primitive(bencher: Bencher, patch_stride: usize) {
    bench_canonicalize(
        bencher,
        make_dict_of_sparse_codes(patch_stride, &PRIMITIVE_VALUES),
    );
}

#[divan::bench(args = PATCH_STRIDES)]
fn canonicalize_dict_sparse_codes_utf8(bencher: Bencher, patch_stride: usize) {
    bench_canonicalize(
        bencher,
        make_dict_of_sparse_codes(patch_stride, &UTF8_VALUES),
    );
}

fn bench_canonicalize(bencher: Bencher, dict: ArrayRef) {
    bencher
        .with_inputs(|| (dict.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .execute::<Canonical>(&mut ctx)
                .vortex_expect("dictionary over sparse codes canonicalization")
        });
}
