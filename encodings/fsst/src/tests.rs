// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use fsst::CompressorBuilder;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::FSST;
use crate::array::FSSTArrayExt;
use crate::compress::fsst_compress_with_max_i32_bytes;
use crate::fsst_compress;
use crate::fsst_train_compressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// this function is VERY slow on miri, so we only want to run it once
pub(crate) fn build_fsst_array(ctx: &mut ExecutionCtx) -> ArrayRef {
    let mut input_array = VarBinBuilder::<i32>::with_capacity_in(
        DType::Utf8(Nullability::NonNullable),
        3,
        BufferAllocatorRef::static_ref(),
    );
    input_array.append_value(b"The Greeks never said that the limit could not be overstepped");
    input_array.append_value(
        b"They said it existed and that whoever dared to exceed it was mercilessly struck down",
    );
    input_array.append_value(b"Nothing in present history can contradict them");
    let input_array = input_array.finish_into_varbin().into_array();

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
    let fsst_sliced = fsst_array
        .slice(1..3)
        .unwrap()
        .optimize_ctx(&SESSION)
        .unwrap();
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

    assert_arrays_eq!(
        fsst_array,
        VarBinViewArray::from_iter_str([
            "The Greeks never said that the limit could not be overstepped",
            "They said it existed and that whoever dared to exceed it was mercilessly struck down",
            "Nothing in present history can contradict them",
        ]),
        &mut ctx
    );
    assert_arrays_eq!(
        fsst_filtered,
        VarBinViewArray::from_iter_str([
            "They said it existed and that whoever dared to exceed it was mercilessly struck down",
            "Nothing in present history can contradict them",
        ]),
        &mut ctx
    );
}

/// Regression for #7833: [`fsst_compress`] must switch its codes offsets from `i32` to `i64`
/// once the worst-case compressed byte count exceeds the `i32` threshold.
///
/// Uses [`fsst_compress_with_max_i32_bytes`] with a small KiB threshold so both the `i32`
/// (at-threshold) and `i64` (above-threshold) paths for `VarBin` and `VarBinView` inputs run in
/// milliseconds without allocating gigabytes. With an empty symbol table every input byte is
/// escape-coded to two output bytes, deterministically crossing the threshold.
#[test]
fn fsst_compress_offsets_overflow_i32() -> VortexResult<()> {
    const MAX_I32_BYTES: usize = 4 * 1024;
    const ROW_LEN: usize = 256;
    const ROWS_FITTING: usize = (MAX_I32_BYTES / 2) / ROW_LEN; // 8 rows -> 2048 input -> 4096 compressed
    const ROWS_OVERFLOWING: usize = ROWS_FITTING + 1; // 9 rows -> 2304 input -> 4608 compressed

    let compressor = CompressorBuilder::default().build();
    let mut ctx = SESSION.create_execution_ctx();
    let row = "a".repeat(ROW_LEN);

    for (n_rows, expected_ptype, exceeds_threshold) in [
        (ROWS_FITTING, PType::I32, false),
        (ROWS_OVERFLOWING, PType::I64, true),
    ] {
        let mut varbin_builder = VarBinBuilder::<i64>::with_capacity_in(
            DType::Utf8(Nullability::NonNullable),
            n_rows,
            BufferAllocatorRef::static_ref(),
        );
        for _ in 0..n_rows {
            varbin_builder.append_value(row.as_bytes());
        }
        let varbin = varbin_builder.finish_into_varbin().into_array();
        let varbinview =
            VarBinViewArray::from_iter_str((0..n_rows).map(|_| row.as_str())).into_array();

        for input in [varbin, varbinview] {
            let compressed =
                fsst_compress_with_max_i32_bytes(&input, &compressor, MAX_I32_BYTES, &mut ctx)?;
            assert_eq!(compressed.len(), n_rows);
            assert_eq!(
                compressed.codes().offsets().dtype().as_ptype(),
                expected_ptype
            );
            assert_eq!(
                compressed.codes_bytes().len() > MAX_I32_BYTES,
                exceeds_threshold
            );
            assert_arrays_eq!(compressed, input, &mut ctx);
        }
    }

    Ok(())
}
