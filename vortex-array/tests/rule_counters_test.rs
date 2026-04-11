// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test to measure parent rule/kernel lookup overhead.

use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::VortexSessionExecute;
use vortex_array::kernel::parent_kernel_counters;
use vortex_array::optimizer::rules::parent_rule_counters;
use vortex_array::optimizer::{ArrayOptimizer, optimizer_counters};
use vortex_array::LEGACY_SESSION;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_array::{Canonical, IntoArray};
use vortex_buffer::Buffer;

fn make_primitive(vals: Vec<i32>) -> vortex_array::ArrayRef {
    PrimitiveArray::new(Buffer::from(vals), Validity::NonNullable).into_array()
}

#[test]
fn measure_rule_overhead_cast_primitive() {
    parent_rule_counters::reset();
    parent_kernel_counters::reset();
    optimizer_counters::reset();

    // Simple case: Cast(Primitive) — one parent, one child
    let arr = PrimitiveArray::new(
        Buffer::from(vec![1i32, 2, 3, 4, 5]),
        Validity::NonNullable,
    )
    .into_array();

    let cast = Cast
        .try_new_array(5, DType::Primitive(vortex_array::dtype::PType::I64, Nullability::NonNullable), [arr])
        .unwrap();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let _result = cast.execute::<Canonical>(&mut ctx).unwrap();

    parent_rule_counters::report("cast_primitive");
    parent_kernel_counters::report("cast_primitive");
    optimizer_counters::report("cast_primitive");
}

#[test]
fn measure_rule_overhead_compare_two_primitives() {
    parent_rule_counters::reset();
    parent_kernel_counters::reset();
    optimizer_counters::reset();

    // Binary compare: Binary(Primitive, Primitive) — one parent, two children
    let a = PrimitiveArray::new(
        Buffer::from(vec![1i32, 2, 3, 4, 5]),
        Validity::NonNullable,
    )
    .into_array();
    let b = PrimitiveArray::new(
        Buffer::from(vec![5i32, 4, 3, 2, 1]),
        Validity::NonNullable,
    )
    .into_array();

    let cmp = Binary
        .try_new_array(5, Operator::Lt, [a, b])
        .unwrap();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let _result = cmp.execute::<Canonical>(&mut ctx).unwrap();

    parent_rule_counters::report("compare_two_primitives");
    parent_kernel_counters::report("compare_two_primitives");
    optimizer_counters::report("compare_two_primitives");
}

#[test]
fn measure_rule_overhead_nested_cast_compare() {
    parent_rule_counters::reset();
    parent_kernel_counters::reset();
    optimizer_counters::reset();

    // Nested: Cast(Binary(Primitive, Primitive)) — deeper tree
    let a = PrimitiveArray::new(
        Buffer::from(vec![1i32, 2, 3, 4, 5]),
        Validity::NonNullable,
    )
    .into_array();
    let b = PrimitiveArray::new(
        Buffer::from(vec![5i32, 4, 3, 2, 1]),
        Validity::NonNullable,
    )
    .into_array();

    let cmp = Binary
        .try_new_array(5, Operator::Lt, [a, b])
        .unwrap();

    // Optimize the comparison first
    let optimized = cmp.optimize().unwrap();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let _result = optimized.execute::<Canonical>(&mut ctx).unwrap();

    parent_rule_counters::report("nested_cast_compare");
    parent_kernel_counters::report("nested_cast_compare");
    optimizer_counters::report("nested_cast_compare");
}

#[test]
fn measure_rule_overhead_chunked_cast() {
    parent_rule_counters::reset();
    parent_kernel_counters::reset();
    optimizer_counters::reset();

    // Chunked array with 10 chunks, each cast to i64
    // This exercises ChunkedUnaryScalarFnPushDownRule
    let chunks: Vec<_> = (0..10).map(|i| make_primitive(vec![i; 100])).collect();
    let chunked = unsafe {
        ChunkedArray::new_unchecked(
            chunks,
            DType::Primitive(vortex_array::dtype::PType::I32, Nullability::NonNullable),
        )
    }
    .into_array();

    let cast = Cast
        .try_new_array(
            1000,
            DType::Primitive(vortex_array::dtype::PType::I64, Nullability::NonNullable),
            [chunked],
        )
        .unwrap();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let _result = cast.execute::<Canonical>(&mut ctx).unwrap();

    parent_rule_counters::report("chunked_cast_10_chunks");
    parent_kernel_counters::report("chunked_cast_10_chunks");
    optimizer_counters::report("chunked_cast_10_chunks");
}

#[test]
fn measure_rule_overhead_many_casts() {
    parent_rule_counters::reset();
    parent_kernel_counters::reset();
    optimizer_counters::reset();

    // Execute 100 separate cast operations to see aggregate overhead
    for _ in 0..100 {
        let arr = make_primitive(vec![1, 2, 3, 4, 5]);
        let cast = Cast
            .try_new_array(
                5,
                DType::Primitive(vortex_array::dtype::PType::I64, Nullability::NonNullable),
                [arr],
            )
            .unwrap();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let _result = cast.execute::<Canonical>(&mut ctx).unwrap();
    }

    parent_rule_counters::report("100_casts");
    parent_kernel_counters::report("100_casts");
    optimizer_counters::report("100_casts");
}

#[test]
fn measure_rule_overhead_filter_primitive() {
    use vortex_mask::Mask;

    parent_rule_counters::reset();
    parent_kernel_counters::reset();
    optimizer_counters::reset();

    let arr = make_primitive((0..1000).collect());
    let mask = Mask::from_iter((0..1000).map(|i| i % 3 == 0));

    let filtered = arr.filter(mask).unwrap();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let _result = filtered.execute::<Canonical>(&mut ctx).unwrap();

    parent_rule_counters::report("filter_primitive");
    parent_kernel_counters::report("filter_primitive");
    optimizer_counters::report("filter_primitive");
}
