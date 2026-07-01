// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_possible_wrap)]
#![expect(clippy::unwrap_used)]

use std::sync::Arc;

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ListViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType::I32;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

const ZCTL_ARGS: &[(usize, usize)] = &[
    // num_lists, list_size
    (1_000, 8),
    (1_000, 64),
    (10_000, 8),
];

const NON_ZCTL_ARGS: &[(usize, usize)] = &[
    // num_lists, list_size
    (1_000, 8),
    (1_000, 32),
    (10_000, 8),
];

fn make_listview(
    num_lists: usize,
    list_size: usize,
    step: usize,
    with_nulls: bool,
) -> ListViewArray {
    let element_count = step * num_lists + list_size;
    let elements = PrimitiveArray::from_iter(0..element_count as i32).into_array();
    let offsets: Buffer<u32> = (0..num_lists).map(|i| (i * step) as u32).collect();
    let sizes: Buffer<u16> = std::iter::repeat_n(list_size as u16, num_lists).collect();
    let validity = if with_nulls {
        Validity::from_iter((0..num_lists).map(|i| i % 5 != 0))
    } else {
        Validity::NonNullable
    };

    ListViewArray::new(elements, offsets.into_array(), sizes.into_array(), validity)
}

#[divan::bench(args = ZCTL_ARGS)]
fn extend_from_array_zctl(bencher: Bencher, (num_lists, list_size): (usize, usize)) {
    let source = make_listview(num_lists, list_size, list_size, false);
    debug_assert!(source.is_zero_copy_to_list());
    let source = source.into_array();

    bencher.with_inputs(|| &source).bench_refs(|source| {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ListViewBuilder::<u64, u64>::with_capacity(
            Arc::new(DType::Primitive(I32, NonNullable)),
            NonNullable,
            num_lists * list_size,
            num_lists,
        );
        source.append_to_builder(&mut builder, &mut ctx).unwrap();
        divan::black_box(builder.finish_into_listview())
    });
}

#[divan::bench(args = NON_ZCTL_ARGS)]
fn extend_from_array_non_zctl_overlapping(
    bencher: Bencher,
    (num_lists, list_size): (usize, usize),
) {
    // `step = 1` creates heavily overlapping lists, which forces the non-ZCTL extend path.
    let source = make_listview(num_lists, list_size, 1, true);
    debug_assert!(!source.is_zero_copy_to_list());
    let source = source.into_array();

    bencher.with_inputs(|| &source).bench_refs(|source| {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = ListViewBuilder::<u64, u64>::with_capacity(
            Arc::new(DType::Primitive(I32, NonNullable)),
            Nullable,
            num_lists * list_size,
            num_lists,
        );
        source.append_to_builder(&mut builder, &mut ctx).unwrap();
        divan::black_box(builder.finish_into_listview())
    });
}
