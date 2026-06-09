// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType::I32;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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

fn make_sparse_list(len: usize, patch_stride: usize, list_size: usize) -> ArrayRef {
    let patch_indices: Buffer<u32> = (0..len).step_by(patch_stride).map(|i| i as u32).collect();
    let n_patches = patch_indices.len();

    let patch_elements = PrimitiveArray::from_iter(0..(n_patches * list_size) as i32).into_array();
    let patch_offsets: Buffer<u32> = (0..n_patches).map(|i| (i * list_size) as u32).collect();
    let patch_sizes: Buffer<u32> = std::iter::repeat_n(list_size as u32, n_patches).collect();
    let patch_values = ListViewArray::new(
        patch_elements,
        patch_offsets.into_array(),
        patch_sizes.into_array(),
        Validity::NonNullable,
    )
    .into_array();

    let fill_value = Scalar::list(
        Arc::new(I32.into()),
        (0..list_size as i32).map(Scalar::from).collect(),
        NonNullable,
    );

    Sparse::try_new(patch_indices.into_array(), patch_values, len, fill_value)
        .vortex_expect("sparse list input should be valid")
        .into_array()
}

fn make_sparse_fixed_size_list(len: usize, patch_stride: usize, list_size: u32) -> ArrayRef {
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

    Sparse::try_new(patch_indices.into_array(), patch_values, len, fill_value)
        .vortex_expect("sparse fixed-size-list input should be valid")
        .into_array()
}

#[divan::bench(args = LIST_ARGS)]
fn canonicalize_sparse_list(
    bencher: Bencher,
    (len, patch_stride, list_size): (usize, usize, usize),
) {
    let sparse = make_sparse_list(len, patch_stride, list_size);

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
