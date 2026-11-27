// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType::I32;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::patches::Patches;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
#[cfg(not(codspeed))]
use vortex_array::test_harness::fusion::fusion_trace;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
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

const LIST_ARGS: &[(usize, usize, usize)] = &[
    // len, patch_stride, list_size
    (512, 7, 4),
    (1_024, 17, 8),
];

const FIXED_SIZE_LIST_ARGS: &[(usize, usize, u32)] = &[
    // len, patch_stride, list_size
    (512, 7, 4),
    (1_024, 17, 8),
];

fn make_sparse_list(
    len: usize,
    patch_stride: usize,
    list_size: usize,
    fill_value: Scalar,
) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let patch_indices: Buffer<u32> = (0..len).step_by(patch_stride).map(|i| i as u32).collect();
    let n_patches = patch_indices.len();

    let patch_elements = PrimitiveArray::from_iter(0..(n_patches * list_size) as i32).into_array();
    let patch_offsets: Buffer<u32> = (0..n_patches).map(|i| (i * list_size) as u32).collect();
    let patch_sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, n_patches).collect();
    let patch_values = ListViewArray::new(
        patch_elements,
        patch_offsets.into_array(),
        patch_sizes.into_array(),
        Validity::from(fill_value.dtype().nullability()),
        &mut ctx,
    )
    .into_array();

    let array = Sparse::try_new(
        patch_indices.into_array(),
        patch_values,
        len,
        fill_value,
        &mut ctx,
    )
    .vortex_expect("sparse list input should be valid")
    .into_array();
    mark_optimized(&array, &SESSION).vortex_expect("mark the sparse list as optimized");

    array
}

fn make_sparse_fixed_size_list(len: usize, patch_stride: usize, list_size: u32) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let patch_indices: Buffer<u32> = (0..len).step_by(patch_stride).map(|i| i as u32).collect();
    let n_patches = patch_indices.len();

    let patch_elements =
        PrimitiveArray::from_iter(0..(n_patches * list_size as usize) as i32).into_array();
    let patch_values =
        FixedSizeListArray::new(patch_elements, list_size, Validity::NonNullable, n_patches)
            .into_array();

    let fill_value = Scalar::fixed_size_list(
        Arc::new(I32.into()),
        (0..list_size as i32).map(Scalar::from).collect(),
        NonNullable,
    );

    let array = Sparse::try_new(
        patch_indices.into_array(),
        patch_values,
        len,
        fill_value,
        &mut ctx,
    )
    .vortex_expect("sparse fixed-size-list input should be valid")
    .into_array();
    mark_optimized(&array, &SESSION).vortex_expect("mark the sparse fixed-size list as optimized");

    array
}

#[divan::bench(args = LIST_ARGS)]
fn canonicalize_sparse_list(
    bencher: Bencher,
    (len, patch_stride, list_size): (usize, usize, usize),
) {
    let fill_value = Scalar::list(
        Arc::new(I32.into()),
        (0..list_size as i32).map(Scalar::from).collect(),
        NonNullable,
    );
    let sparse = make_sparse_list(len, patch_stride, list_size, fill_value);

    bench_sparse_list(bencher, sparse);
}

#[divan::bench(args = LIST_ARGS)]
fn canonicalize_sparse_list_null_fill(
    bencher: Bencher,
    (len, patch_stride, list_size): (usize, usize, usize),
) {
    // Present patches are valid lists; only the rows between patches are null.
    let fill_value = Scalar::null(DType::List(Arc::new(I32.into()), Nullable));
    let sparse = make_sparse_list(len, patch_stride, list_size, fill_value);

    bench_sparse_list(bencher, sparse);
}

fn bench_sparse_list(bencher: Bencher, sparse: ArrayRef) {
    bencher
        .with_inputs(|| (sparse.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(
                array
                    .execute::<Canonical>(&mut ctx)
                    .vortex_expect("sparse list canonicalization"),
            )
        });
}

#[divan::bench(args = FIXED_SIZE_LIST_ARGS)]
fn canonicalize_sparse_fixed_size_list(
    bencher: Bencher,
    (len, patch_stride, list_size): (usize, usize, u32),
) {
    let sparse = make_sparse_fixed_size_list(len, patch_stride, list_size);

    bencher
        .with_inputs(|| (sparse.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            divan::black_box(
                array
                    .execute::<Canonical>(&mut ctx)
                    .vortex_expect("sparse fixed-size-list canonicalization"),
            )
        });
}

/// Sparse counters with 1% or 10% observations, optionally missing between observations.
#[divan::bench(args = [
    (100, false, false), (10, false, false), (100, true, false), (10, true, false),
    (100, false, true), (10, false, true), (100, true, true), (10, true, true),
])]
fn primitive_patches(bencher: Bencher, (stride, null_fill, sliced): (usize, bool, bool)) {
    let len = 65_536u32;
    let mut ctx = SESSION.create_execution_ctx();
    let indices = PrimitiveArray::from_iter((0..len).step_by(stride));
    let observed = (0..indices.len()).map(|i| (i as i64) * 7 + 12);
    let (values, fill) = if null_fill {
        (
            PrimitiveArray::from_option_iter(observed.map(Some)),
            Scalar::null_native::<i64>(),
        )
    } else {
        (PrimitiveArray::from_iter(observed), Scalar::from(12i64))
    };
    let array = Sparse::try_new(
        indices.into_array(),
        values.into_array(),
        len as usize,
        fill,
        &mut ctx,
    )
    .vortex_expect("valid sparse counters")
    .into_array();
    let array = if sliced {
        array
            .slice(17..len as usize - 9)
            .vortex_expect("slice sparse counters")
    } else {
        array
    }
    .optimize_ctx(&SESSION)
    .vortex_expect("optimize sparse counters");
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(array, mut ctx)| {
            array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("decode sparse counters")
        });
}

/// A sparse sensor channel and a second, independently recorded set of corrections.
#[divan::bench(args = [false, true])]
fn patched_sparse_readings(bencher: Bencher, fusion: bool) {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = 65_536u32;
    let indices = (0..rows).step_by(101);
    let sparse = Sparse::try_new(
        PrimitiveArray::from_iter(indices.clone()).into_array(),
        PrimitiveArray::from_iter(indices.map(|row| row + 500)).into_array(),
        rows as usize,
        Scalar::from(50u32),
        &mut ctx,
    )
    .vortex_expect("sparse readings fixture");
    let corrections = (0..rows).step_by(197);
    let patches = Patches::new(
        rows as usize,
        0,
        PrimitiveArray::from_iter(corrections.clone()).into_array(),
        PrimitiveArray::from_iter(corrections.map(|row| row + 900)).into_array(),
        None,
        &mut ctx,
    )
    .vortex_expect("corrections fixture");
    let array = Patched::from_array_and_patches(sparse.into_array(), &patches, &mut ctx)
        .vortex_expect("patched sparse readings")
        .into_array()
        .slice(31..rows as usize - 49)
        .vortex_expect("unaligned sensor window");
    #[cfg(not(codspeed))]
    {
        let trace =
            fusion_trace(array.clone(), &SESSION).vortex_expect("blockwise and full decode agree");
        assert!(
            !trace.contains("fuse extend parent=vortex.patched"),
            "{trace}"
        );
    }
    bencher
        .with_inputs(|| {
            (
                array.clone(),
                if fusion {
                    SESSION.create_execution_ctx().with_forced_block_fusion()
                } else {
                    SESSION.create_execution_ctx().with_block_fusion(false)
                },
            )
        })
        .bench_values(|(array, mut ctx)| {
            array
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("decode readings")
        });
}
