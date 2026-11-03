// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversions between list scalars and arrays: reading a list row as a scalar, expanding a list
//! scalar into an array, and building, comparing and serializing list scalars.

#![expect(clippy::unwrap_used)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<KernelSession>());

/// Elements per list.
const LIST_SIZES: &[usize] = &[4, 64, 1024];
/// Rows in the source list arrays.
const NUM_ROWS: usize = 256;
/// Rows read per iteration.
const NUM_READS: usize = 64;
/// Rows a constant list array expands to.
const CONSTANT_LEN: usize = 1024;

fn i64_list_array(list_size: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let elements = (0..(NUM_ROWS * list_size) as i64)
        .collect::<Buffer<i64>>()
        .into_array();
    let offsets = (0..NUM_ROWS as u32)
        .map(|i| i * list_size as u32)
        .collect::<Buffer<u32>>()
        .into_array();
    let sizes = std::iter::repeat_n(list_size as u32, NUM_ROWS)
        .collect::<Buffer<u32>>()
        .into_array();
    ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx).into_array()
}

fn utf8_list_array(list_size: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let elements = VarBinViewArray::from_iter_str(
        (0..NUM_ROWS * list_size).map(|i| format!("value-{i}-with-a-longer-suffix")),
    )
    .into_array();
    let offsets = (0..NUM_ROWS as u32)
        .map(|i| i * list_size as u32)
        .collect::<Buffer<u32>>()
        .into_array();
    let sizes = std::iter::repeat_n(list_size as u32, NUM_ROWS)
        .collect::<Buffer<u32>>()
        .into_array();
    ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx).into_array()
}

fn nested_list_array(list_size: usize) -> ArrayRef {
    // list<list<i64>>, each outer row holding `list_size` inner lists of 4 elements.
    let mut ctx = SESSION.create_execution_ctx();
    let inner_rows = NUM_ROWS * list_size;
    let inner = {
        let elements = (0..(inner_rows * 4) as i64)
            .collect::<Buffer<i64>>()
            .into_array();
        let offsets = (0..inner_rows as u32)
            .map(|i| i * 4)
            .collect::<Buffer<u32>>()
            .into_array();
        let sizes = std::iter::repeat_n(4u32, inner_rows)
            .collect::<Buffer<u32>>()
            .into_array();
        ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx).into_array()
    };
    let offsets = (0..NUM_ROWS as u32)
        .map(|i| i * list_size as u32)
        .collect::<Buffer<u32>>()
        .into_array();
    let sizes = std::iter::repeat_n(list_size as u32, NUM_ROWS)
        .collect::<Buffer<u32>>()
        .into_array();
    ListViewArray::new(inner, offsets, sizes, Validity::NonNullable, &mut ctx).into_array()
}

/// `list<struct{a: i64, b: i64}>` whose fields are dictionary-encoded, as a compressed column's
/// fields would be.
fn struct_dict_list_array(list_size: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let len = NUM_ROWS * list_size;
    let dict_field = |seed: u64| {
        let codes = (0..len as u64)
            .map(|i| ((i * 7 + seed) % 16) as u8)
            .collect::<Buffer<u8>>()
            .into_array();
        let values = (0..16i64)
            .map(|v| v * 1_000)
            .collect::<Buffer<i64>>()
            .into_array();
        DictArray::try_new(codes, values).unwrap().into_array()
    };
    let elements = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        [dict_field(1), dict_field(5)],
        len,
        Validity::NonNullable,
    )
    .unwrap()
    .into_array();
    let offsets = (0..NUM_ROWS as u32)
        .map(|i| i * list_size as u32)
        .collect::<Buffer<u32>>()
        .into_array();
    let sizes = std::iter::repeat_n(list_size as u32, NUM_ROWS)
        .collect::<Buffer<u32>>()
        .into_array();
    ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx).into_array()
}

fn f32_fsl_array(list_size: usize) -> ArrayRef {
    let elements = PrimitiveArray::from_iter((0..NUM_ROWS * list_size).map(|i| i as f32));
    FixedSizeListArray::new(
        elements.into_array(),
        list_size as u32,
        Validity::NonNullable,
        NUM_ROWS,
    )
    .into_array()
}

fn read_rows(array: &ArrayRef) -> Vec<Scalar> {
    let mut ctx = SESSION.create_execution_ctx();
    (0..NUM_READS)
        .map(|i| array.execute_scalar(i * 3 % NUM_ROWS, &mut ctx).unwrap())
        .collect()
}

fn i64_list_scalar(list_size: usize) -> Scalar {
    Scalar::list(
        Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
        (0..list_size as i64)
            .map(|i| Scalar::primitive(i, Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    )
}

// ---------------------------------------------------------------------------------------------
// Array -> scalar
// ---------------------------------------------------------------------------------------------

#[divan::bench(args = LIST_SIZES)]
fn array_to_scalar_list_i64(bencher: Bencher, list_size: usize) {
    let array = i64_list_array(list_size);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for i in 0..NUM_READS {
                divan::black_box(array.execute_scalar(i * 3 % NUM_ROWS, ctx).unwrap());
            }
        });
}

#[divan::bench(args = LIST_SIZES)]
fn array_to_scalar_list_utf8(bencher: Bencher, list_size: usize) {
    let array = utf8_list_array(list_size);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for i in 0..NUM_READS {
                divan::black_box(array.execute_scalar(i * 3 % NUM_ROWS, ctx).unwrap());
            }
        });
}

#[divan::bench(args = LIST_SIZES)]
fn array_to_scalar_list_nested(bencher: Bencher, list_size: usize) {
    let array = nested_list_array(list_size);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for i in 0..NUM_READS {
                divan::black_box(array.execute_scalar(i * 3 % NUM_ROWS, ctx).unwrap());
            }
        });
}

#[divan::bench(args = LIST_SIZES)]
fn array_to_scalar_list_struct_dict(bencher: Bencher, list_size: usize) {
    let array = struct_dict_list_array(list_size);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for i in 0..NUM_READS {
                divan::black_box(array.execute_scalar(i * 3 % NUM_ROWS, ctx).unwrap());
            }
        });
}

#[divan::bench(args = LIST_SIZES)]
fn array_to_scalar_fsl_f32(bencher: Bencher, list_size: usize) {
    let array = f32_fsl_array(list_size);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for i in 0..NUM_READS {
                divan::black_box(array.execute_scalar(i * 3 % NUM_ROWS, ctx).unwrap());
            }
        });
}

// ---------------------------------------------------------------------------------------------
// Scalar -> array
// ---------------------------------------------------------------------------------------------

fn bench_constant_canonical(bencher: Bencher, scalars: Vec<Scalar>) {
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for scalar in &scalars {
                let array = ConstantArray::new(scalar.clone(), CONSTANT_LEN).into_array();
                divan::black_box(array.execute::<Canonical>(ctx).unwrap());
            }
        });
}

#[divan::bench(args = LIST_SIZES)]
fn scalar_to_array_list_i64(bencher: Bencher, list_size: usize) {
    bench_constant_canonical(bencher, read_rows(&i64_list_array(list_size)));
}

#[divan::bench(args = LIST_SIZES)]
fn scalar_to_array_list_utf8(bencher: Bencher, list_size: usize) {
    bench_constant_canonical(bencher, read_rows(&utf8_list_array(list_size)));
}

#[divan::bench(args = LIST_SIZES)]
fn scalar_to_array_list_nested(bencher: Bencher, list_size: usize) {
    bench_constant_canonical(bencher, read_rows(&nested_list_array(list_size)));
}

#[divan::bench(args = LIST_SIZES)]
fn scalar_to_array_list_struct_dict(bencher: Bencher, list_size: usize) {
    bench_constant_canonical(bencher, read_rows(&struct_dict_list_array(list_size)));
}

/// Reads every element of list scalars whose struct elements are dictionary-encoded, the cost a
/// scalar's lazily held children move to.
#[divan::bench(args = LIST_SIZES)]
fn read_elements_list_struct_dict(bencher: Bencher, list_size: usize) {
    let scalars = read_rows(&struct_dict_list_array(list_size));
    bencher.bench(|| {
        for scalar in &scalars {
            divan::black_box(scalar.as_list().elements());
        }
    });
}

#[divan::bench(args = LIST_SIZES)]
fn scalar_to_array_fsl_f32(bencher: Bencher, list_size: usize) {
    bench_constant_canonical(bencher, read_rows(&f32_fsl_array(list_size)));
}

/// Appends list scalars to a builder, as `take`/`interleave`/fill-null paths over constants do.
#[divan::bench(args = LIST_SIZES)]
fn scalar_to_array_builder_append(bencher: Bencher, list_size: usize) {
    let scalars = read_rows(&i64_list_array(list_size));
    let dtype = scalars[0].dtype().clone();
    bencher.bench(|| {
        let mut builder =
            builder_with_capacity_in(&dtype, scalars.len(), BufferAllocatorRef::static_ref());
        for scalar in &scalars {
            builder.append_scalar(scalar).unwrap();
        }
        divan::black_box(builder.finish())
    });
}

/// Reads a list row and expands it back into a constant array, end to end.
#[divan::bench(args = LIST_SIZES)]
fn round_trip_list_i64(bencher: Bencher, list_size: usize) {
    let array = i64_list_array(list_size);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for i in 0..NUM_READS {
                let scalar = array.execute_scalar(i * 3 % NUM_ROWS, ctx).unwrap();
                let constant = ConstantArray::new(scalar, CONSTANT_LEN).into_array();
                divan::black_box(constant.execute::<Canonical>(ctx).unwrap());
            }
        });
}

// ---------------------------------------------------------------------------------------------
// Scalar operations
// ---------------------------------------------------------------------------------------------

#[divan::bench(args = LIST_SIZES)]
fn construct_list_scalar(bencher: Bencher, list_size: usize) {
    let element_dtype = Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable));
    bencher
        .with_inputs(|| {
            (0..list_size as i64)
                .map(|i| Scalar::primitive(i, Nullability::NonNullable))
                .collect::<Vec<_>>()
        })
        .bench_values(|children| {
            Scalar::list(
                Arc::clone(&element_dtype),
                children,
                Nullability::NonNullable,
            )
        });
}

#[divan::bench(args = LIST_SIZES)]
fn eq_list_scalar(bencher: Bencher, list_size: usize) {
    let lhs = i64_list_scalar(list_size);
    let rhs = i64_list_scalar(list_size);
    bencher.bench(|| divan::black_box(&lhs) == divan::black_box(&rhs));
}

#[divan::bench(args = LIST_SIZES)]
fn hash_list_scalar(bencher: Bencher, list_size: usize) {
    let scalar = i64_list_scalar(list_size);
    let state = RandomState::new();
    bencher.bench(|| state.hash_one(divan::black_box(&scalar)));
}

#[divan::bench(args = LIST_SIZES)]
fn cmp_list_scalar(bencher: Bencher, list_size: usize) {
    let lhs = i64_list_scalar(list_size);
    let rhs = i64_list_scalar(list_size);
    bencher.bench(|| divan::black_box(&lhs).partial_cmp(divan::black_box(&rhs)));
}

#[divan::bench(args = LIST_SIZES)]
fn list_scalar_element_access(bencher: Bencher, list_size: usize) {
    let scalar = i64_list_scalar(list_size);
    bencher.bench(|| {
        let list = scalar.as_list();
        for i in (0..list_size).step_by(list_size.div_ceil(16)) {
            divan::black_box(list.element(i));
        }
    });
}

#[divan::bench(args = LIST_SIZES)]
fn serialize_list_scalar(bencher: Bencher, list_size: usize) {
    let scalar = i64_list_scalar(list_size);
    bencher.bench(|| ScalarValue::to_proto_bytes::<Vec<u8>>(divan::black_box(scalar.value())));
}

#[divan::bench(args = LIST_SIZES)]
fn deserialize_list_scalar(bencher: Bencher, list_size: usize) {
    let scalar = i64_list_scalar(list_size);
    let bytes = ScalarValue::to_proto_bytes::<Vec<u8>>(scalar.value());
    bencher.bench(|| {
        ScalarValue::from_proto_bytes(divan::black_box(&bytes), scalar.dtype(), &SESSION).unwrap()
    });
}

#[divan::bench(args = LIST_SIZES)]
fn cast_list_scalar(bencher: Bencher, list_size: usize) {
    let scalar = i64_list_scalar(list_size);
    let target = DType::List(
        Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
        Nullability::Nullable,
    );
    bencher.bench(|| divan::black_box(&scalar).cast(&target).unwrap());
}
