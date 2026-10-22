// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar reads through one-off and repeated probes, over arrays whose validity is a child's
//! (Dict over nullable values), an encoding's own (nullable Primitive), or a slot's (nullable
//! Struct). Rows are read in a fixed pseudo-random order that lands on nulls as well as values.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const LEN: usize = 65_536;
const VALUES: usize = 1_024;
// Sized to keep the CodSpeed simulation short per benchmark.
const ACCESSES: usize = 256;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<KernelSession>());

fn nullable_primitive(len: usize) -> ArrayRef {
    PrimitiveArray::new(
        (0..len)
            .map(|i| i64::try_from(i).vortex_expect("fixture values fit i64"))
            .collect::<Buffer<i64>>(),
        Validity::from_iter((0..len).map(|i| i % 7 != 0)),
    )
    .into_array()
}

fn dict_over_nullable_values() -> ArrayRef {
    let codes = (0..LEN)
        .map(|i| u16::try_from((i * 7919) % VALUES).vortex_expect("codes fit u16"))
        .collect::<Buffer<u16>>()
        .into_array();
    DictArray::try_new(codes, nullable_primitive(VALUES))
        .vortex_expect("dict fixture")
        .into_array()
}

fn nullable_struct() -> ArrayRef {
    StructArray::try_new(
        FieldNames::from(["a", "b"]),
        vec![nullable_primitive(LEN), nullable_primitive(LEN)],
        LEN,
        Validity::from_iter((0..LEN).map(|i| i % 13 != 0)),
    )
    .vortex_expect("struct fixture")
    .into_array()
}

fn nullable_map() -> ArrayRef {
    let entries_len = LEN * 3;
    let keys = PrimitiveArray::from_iter(
        (0..entries_len)
            // Each group of three entries has keys 0, 1, 0.
            .map(|i| i32::try_from(i % 3 % 2).vortex_expect("key fits i32")),
    )
    .into_array();
    let values = VarBinViewArray::from_iter(
        [Some("short"), None, Some("a longer map value")]
            .into_iter()
            .cycle()
            .take(entries_len),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();
    let entries = StructArray::try_new(
        FieldNames::from(["key", "value"]),
        vec![keys, values],
        entries_len,
        Validity::NonNullable,
    )
    .vortex_expect("map entries fixture")
    .into_array();
    let offsets = (0..LEN)
        .map(|i| u32::try_from(i * 3).vortex_expect("map offset fits u32"))
        .collect::<Buffer<u32>>()
        .into_array();
    let sizes = (0..LEN)
        .map(|i| u32::try_from(i % 4).vortex_expect("map size fits u32"))
        .collect::<Buffer<u32>>()
        .into_array();
    let listview = ListViewArray::new(
        entries,
        offsets,
        sizes,
        Validity::from_iter((0..LEN).map(|i| i % 11 != 0)),
        &mut SESSION.create_execution_ctx(),
    );
    let dtype = MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        false,
    )
    .vortex_expect("map dtype fixture");
    MapArray::try_new(dtype, listview)
        .vortex_expect("map fixture")
        .into_array()
}

fn scalar_fn(nested: bool) -> ArrayRef {
    let add = TypedScalarFnInstance::new(Binary, Operator::Add).erased();
    let lhs = nullable_primitive(LEN);
    let rhs = ConstantArray::new(3i64, LEN).into_array();
    let added = ScalarFnArray::try_new(add, vec![lhs, rhs])
        .vortex_expect("scalar function add fixture")
        .into_array();
    if !nested {
        return added;
    }

    let mul = TypedScalarFnInstance::new(Binary, Operator::Mul).erased();
    ScalarFnArray::try_new(mul, vec![added, nullable_primitive(LEN)])
        .vortex_expect("nested scalar function fixture")
        .into_array()
}

fn indices() -> Vec<usize> {
    let mut seed = 42u64;
    (0..ACCESSES)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 32) as usize % LEN
        })
        .collect()
}

const ARRAYS: &[&str] = &[
    "primitive",
    "dict",
    "struct",
    "map",
    "scalar_fn_binary",
    "scalar_fn_nested",
];

const SCALAR_FN_ARRAYS: &[&str] = &["scalar_fn_binary", "scalar_fn_nested"];

fn array(name: &str) -> ArrayRef {
    match name {
        "primitive" => nullable_primitive(LEN),
        "map" => nullable_map(),
        "scalar_fn_binary" => scalar_fn(false),
        "scalar_fn_nested" => scalar_fn(true),
        "dict" => dict_over_nullable_values(),
        "struct" => nullable_struct(),
        _ => unreachable!("unknown fixture {name}"),
    }
}

#[divan::bench(args = ARRAYS)]
fn once(bencher: Bencher, name: &str) {
    let array = array(name);
    let indices = indices();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for &index in &indices {
                divan::black_box(
                    array
                        .execute_scalar(index, ctx)
                        .vortex_expect("in-bounds read"),
                );
            }
        });
}

#[divan::bench(args = ARRAYS)]
fn repeated(bencher: Bencher, name: &str) {
    let array = array(name);
    let indices = indices();
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for &index in &indices {
                divan::black_box(
                    probe
                        .execute_scalar(index, ctx)
                        .vortex_expect("in-bounds read"),
                );
            }
        });
}

#[divan::bench(args = SCALAR_FN_ARRAYS)]
fn scalar_fn_same_row_once(bencher: Bencher, name: &str) {
    let array = array(name);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for _ in 0..ACCESSES {
                divan::black_box(
                    array
                        .execute_scalar(LEN / 2, ctx)
                        .vortex_expect("in-bounds read"),
                );
            }
        });
}

#[divan::bench(args = SCALAR_FN_ARRAYS)]
fn scalar_fn_same_row_repeated(bencher: Bencher, name: &str) {
    let array = array(name);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for _ in 0..ACCESSES {
                divan::black_box(
                    probe
                        .execute_scalar(LEN / 2, ctx)
                        .vortex_expect("in-bounds read"),
                );
            }
        });
}
