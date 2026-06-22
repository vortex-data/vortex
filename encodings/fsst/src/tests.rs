// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use fsst::CompressorBuilder;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::buffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::FSST;
use crate::fsst_compress;
use crate::fsst_train_compressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// this function is VERY slow on miri, so we only want to run it once
pub(crate) fn build_fsst_array(ctx: &mut ExecutionCtx) -> ArrayRef {
    let mut input_array = VarBinBuilder::<i32>::with_capacity(3);
    input_array.append_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.append_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.append_value(b"Nothing in present history can contradict them");
    let input_array = input_array
        .finish(DType::Utf8(Nullability::NonNullable))
        .into_array();

    let compressor = fsst_train_compressor(&input_array, ctx).unwrap();
    fsst_compress(&input_array, &compressor, ctx)
        .unwrap()
        .into_array()
}

#[test]
fn test_fsst_array_ops() {
    let mut ctx = SESSION.create_execution_ctx();
    // first test the scalar_at values
    let fsst_array = build_fsst_array(&mut ctx);
    assert_nth_scalar!(
        fsst_array,
        0,
        "The Greeks never said that the limit could not be overstepped",
        &mut ctx
    );
    assert_nth_scalar!(
        fsst_array,
        1,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down",
        &mut ctx
    );
    assert_nth_scalar!(
        fsst_array,
        2,
        "Nothing in present history can contradict them",
        &mut ctx
    );

    // test slice
    let fsst_sliced = fsst_array.slice(1..3).unwrap();
    assert!(fsst_sliced.is::<FSST>());
    assert_eq!(fsst_sliced.len(), 2);
    assert_nth_scalar!(
        fsst_sliced,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down",
        &mut ctx
    );
    assert_nth_scalar!(
        fsst_sliced,
        1,
        "Nothing in present history can contradict them",
        &mut ctx
    );

    // test take
    let indices = buffer![0, 2].into_array();
    let fsst_taken = fsst_array.take(indices).unwrap();
    assert_eq!(fsst_taken.len(), 2);
    assert_nth_scalar!(
        fsst_taken,
        0,
        "The Greeks never said that the limit could not be overstepped",
        &mut ctx
    );
    assert_nth_scalar!(
        fsst_taken,
        1,
        "Nothing in present history can contradict them",
        &mut ctx
    );

    // test filter
    let mask = Mask::from_iter([false, true, true]);

    let fsst_filtered = fsst_array.filter(mask).unwrap();

    assert_eq!(fsst_filtered.len(), 2);
    assert_nth_scalar!(
        fsst_filtered,
        0,
        "They said it existed and that whoever dared to exceed it was mercilessly struck down",
        &mut ctx
    );

    // test to_canonical
    let canonical_array = fsst_array
        .clone()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap()
        .into_array();

    assert_arrays_eq!(fsst_array, canonical_array, &mut ctx);
}

// TODO(someone): ideally CI would run this in release mode as well since debug builds make the
// allocation and compression loop substantially slower.
/// Regression for #7833: [`fsst_compress`] must accept inputs whose cumulative compressed
/// bytes exceed [`i32::MAX`]. Before the fix, the compress path hardcoded
/// [`VarBinBuilder<i32>`] for the FSST output and panicked in
/// [`VarBinBuilder::append_value`] once cumulative compressed bytes crossed the boundary.
///
/// The compressor is built with an empty symbol table, so every input byte misses the table
/// and is escape-coded at exactly two output bytes per input byte. This makes the output
/// size deterministic and crosses the i32 boundary with roughly half the input bytes that
/// an incompressible-input construction would need. The input is built with
/// [`VarBinBuilder<i64>`] so the input itself does not panic, which confirms the overflow
/// is on the FSST output side. After the fix the test must succeed with the row count
/// preserved.
///
/// Allocates ~1.1 GiB for the input and ~2.1 GiB for the FSST output (~3.2 GiB total), so
/// it is gated to CI runs and skipped when `VORTEX_SKIP_SLOW_TESTS` is set. To run it
/// locally:
///
/// ```text
/// CI=1 cargo test --release -p vortex-fsst fsst_compress_offsets
/// ```
///
/// [`fsst_compress`]: crate::compress::fsst_compress
#[test_with::env(CI)]
#[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
fn fsst_compress_offsets_overflow_i32() {
    const STRING_LEN: usize = 64 * 1024;
    // Escape coding doubles every byte, so ~1.06 GiB of input compresses to ~2.13 GiB,
    // comfortably past i32::MAX.
    const TOTAL_BYTES: usize = (1usize << 30) + (64 << 20);
    const N: usize = TOTAL_BYTES / STRING_LEN;

    println!("building large VarBinArray");
    let string = vec![b'a'; STRING_LEN];
    let mut builder = VarBinBuilder::<i64>::with_capacity(N);
    for _ in 0..N {
        builder.append_value(&string);
    }
    let array = builder
        .finish(DType::Utf8(Nullability::NonNullable))
        .into_array();

    let compressor = CompressorBuilder::default().build();
    let len = array.len();
    let mut ctx = SESSION.create_execution_ctx();

    println!("compressing to FSST");
    let compressed = fsst_compress(&array, &compressor, &mut ctx).unwrap();
    assert_eq!(compressed.len(), len);
    // Prove the regression condition was exercised: compressed bytes crossed i32::MAX.
    assert!(compressed.codes_bytes().len() > i32::MAX as usize);
}
