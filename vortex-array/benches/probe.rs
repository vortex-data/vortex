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
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VariantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
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

fn nullable_list() -> ArrayRef {
    let offsets = (0..=LEN)
        .map(|i| u32::try_from(i * 3).vortex_expect("list offsets fit u32"))
        .collect::<Buffer<u32>>()
        .into_array();
    ListArray::try_new(
        nullable_primitive(LEN * 3),
        offsets,
        Validity::from_iter((0..LEN).map(|i| i % 11 != 0)),
        &mut SESSION.create_execution_ctx(),
    )
    .vortex_expect("list fixture")
    .into_array()
}

fn nullable_listview() -> ArrayRef {
    let offsets = PrimitiveArray::from_iter(
        (0..LEN).map(|index| u32::try_from(index * 3).vortex_expect("offset fits u32")),
    )
    .into_array();
    let sizes = ConstantArray::new(3u32, LEN).into_array();
    ListViewArray::new(
        nullable_primitive(LEN * 3),
        offsets,
        sizes,
        Validity::from_iter((0..LEN).map(|index| index % 11 != 0)),
        &mut SESSION.create_execution_ctx(),
    )
    .into_array()
}

fn nullable_fixed_size_list() -> ArrayRef {
    FixedSizeListArray::new(
        nullable_primitive(LEN * 3),
        3,
        Validity::from_iter((0..LEN).map(|index| index % 11 != 0)),
        LEN,
    )
    .into_array()
}

fn nullable_filter() -> ArrayRef {
    FilterArray::new(
        nullable_primitive(LEN * 2),
        Mask::from_iter((0..LEN * 2).map(|index| index % 2 == 0)),
    )
    .into_array()
}

fn nullable_union() -> ArrayRef {
    let type_ids = PrimitiveArray::from_option_iter((0..LEN).map(|index| {
        (index % 13 != 0).then_some(if index % 2 == 0 { 5u8 } else { 9u8 })
    }))
    .into_array();
    let variants = UnionVariants::try_new(
        ["number", "flag"].into(),
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
        vec![5, 9],
    )
    .vortex_expect("union variants");
    UnionArray::try_new(
        type_ids,
        variants,
        vec![
            PrimitiveArray::from_iter((0..LEN).map(|index| index as i32)).into_array(),
            BoolArray::from_iter((0..LEN).map(|index| index % 2 == 1)).into_array(),
        ],
    )
    .vortex_expect("union fixture")
    .into_array()
}

fn variant() -> ArrayRef {
    let storage = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), LEN).into_array();
    VariantArray::try_new(storage, None)
        .vortex_expect("variant fixture")
        .into_array()
}

fn nullable_interleave() -> ArrayRef {
    let array_indices = (0..LEN)
        .map(|i| u8::try_from(i % 2).vortex_expect("branch index fits u8"))
        .collect::<Buffer<u8>>()
        .into_array();
    let row_indices = (0..LEN)
        .map(|i| u32::try_from((i * 7919) % LEN).vortex_expect("row index fits u32"))
        .collect::<Buffer<u32>>()
        .into_array();
    InterleaveArray::try_new(
        vec![nullable_primitive(LEN), nullable_primitive(LEN)],
        array_indices,
        row_indices,
    )
    .vortex_expect("interleave fixture")
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
    "varbin",
    "list",
    "listview",
    "fixed_size_list",
    "filter",
    "interleave",
    "union",
    "variant",
    "scalar_fn_validity",
];

fn array(name: &str) -> ArrayRef {
    match name {
        "primitive" => nullable_primitive(LEN),
        "list" => nullable_list(),
        "listview" => nullable_listview(),
        "fixed_size_list" => nullable_fixed_size_list(),
        "filter" => nullable_filter(),
        "union" => nullable_union(),
        "variant" => variant(),
        "scalar_fn_validity" => scalar_fn(false)
            .validity()
            .vortex_expect("scalar function validity")
            .to_array(LEN),
        "interleave" => nullable_interleave(),
        "dict" => dict_over_nullable_values(),
        "struct" => nullable_struct(),
        "varbin" => VarBinArray::from_iter(
            [Some("short"), None, Some("longer varbin string")]
                .into_iter()
                .cycle()
                .take(LEN),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array(),
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
