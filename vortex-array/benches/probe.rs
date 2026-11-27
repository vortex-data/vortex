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
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::patched::PatchedSlots;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::patches::Patches;
use vortex_array::scalar::NumericOperator;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::builtin::boolean::Boolean;
use vortex_array::scalar_fn::builtin::numeric::Numeric;
use vortex_array::scalar_fn::builtin::operators::BooleanOperator;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const LEN: usize = 65_536;
const VALUES: usize = 1_024;
const PIECES: usize = LEN / 8;
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
    let add = TypedScalarFnInstance::new(Numeric, NumericOperator::Add).erased();
    let lhs = nullable_primitive(LEN);
    let rhs = ConstantArray::new(3i64, LEN).into_array();
    let added = ScalarFnArray::try_new(add, vec![lhs, rhs])
        .vortex_expect("scalar function add fixture")
        .into_array();
    if !nested {
        return added;
    }

    let mul = TypedScalarFnInstance::new(Numeric, NumericOperator::Mul).erased();
    ScalarFnArray::try_new(mul, vec![added, nullable_primitive(LEN)])
        .vortex_expect("nested scalar function fixture")
        .into_array()
}

fn patched() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = PrimitiveArray::from_iter(
        (0..LEN).map(|i| u32::try_from(i).vortex_expect("fixture index fits u32")),
    )
    .into_array();
    let len = u32::try_from(LEN).vortex_expect("fixture length fits u32");
    let patch_indices = PrimitiveArray::from_iter((0..len).step_by(8)).into_array();
    let patch_values = PrimitiveArray::from_iter((0..LEN / 8).map(|_| u32::MAX)).into_array();
    let patches = Patches::new(LEN, 0, patch_indices, patch_values, None, &mut ctx)
        .vortex_expect("patched fixture patches");
    Patched::from_array_and_patches(inner, &patches, &mut ctx)
        .vortex_expect("patched fixture")
        .into_array()
}

fn patched_with_chunked_indices() -> ArrayRef {
    let array = patched();
    let patched = array.as_opt::<Patched>().vortex_expect("Patched fixture");
    let indices = patched.patch_indices();
    let middle = indices.len() / 2;
    let chunks = [
        indices
            .slice(0..middle)
            .and_then(|chunk| chunk.optimize_ctx(&*SESSION))
            .vortex_expect("first index chunk"),
        indices
            .slice(middle..indices.len())
            .and_then(|chunk| chunk.optimize_ctx(&*SESSION))
            .vortex_expect("second index chunk"),
    ];
    let chunked = ChunkedArray::try_new(chunks, indices.dtype().clone())
        .vortex_expect("chunked patch indices")
        .into_array();
    let slots = PatchedSlots {
        inner: patched.inner().clone(),
        patch_chunk_offsets: patched.patch_chunk_offsets().clone(),
        patch_indices: chunked,
        patch_values: patched.patch_values().clone(),
    };
    // SAFETY: The chunked child has the same dtype and values as the original index child.
    unsafe { array.with_slots(slots.into_slots()) }.vortex_expect("patched with chunked indices")
}

fn piecewise(variable_lengths: bool) -> ArrayRef {
    let starts = PrimitiveArray::from_iter(
        (0..PIECES).map(|i| u64::try_from(i * 16).vortex_expect("piece start fits u64")),
    )
    .into_array();
    let lengths = if variable_lengths {
        PrimitiveArray::from_iter((0..PIECES).map(|i| if i % 2 == 0 { 4u16 } else { 12u16 }))
            .into_array()
    } else {
        ConstantArray::new(8u16, PIECES).into_array()
    };
    let multipliers = ConstantArray::new(1u8, PIECES).into_array();
    PiecewiseSequenceArray::try_new(starts, lengths, multipliers, LEN)
        .vortex_expect("piecewise fixture")
        .into_array()
}

fn same_piece_indices(variable_lengths: bool) -> Vec<usize> {
    let length = if variable_lengths { 4 } else { 8 };
    let count = if variable_lengths {
        accesses("piecewise_variable")
    } else {
        ACCESSES
    };
    (0..count).map(|i| LEN / 2 + i % length).collect()
}

fn same_lane_indices() -> Vec<usize> {
    (0..ACCESSES).map(|i| (i % 32) * 32).collect()
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

/// Kleene AND is not strict, so its validity is irreducible and reads through
/// `ScalarFnValidity`.
fn kleene_and_validity() -> ArrayRef {
    let lhs = BoolArray::from_iter((0..LEN).map(|i| (i % 7 != 0).then_some(i % 2 == 0)));
    let rhs = BoolArray::from_iter((0..LEN).map(|i| (i % 5 != 0).then_some(i % 3 == 0)));
    Boolean::try_new(lhs.into_array(), rhs.into_array(), BooleanOperator::And)
        .vortex_expect("Kleene AND fixture")
        .into_array()
        .validity()
        .vortex_expect("scalar function validity")
        .to_array(LEN)
}

/// Reads per iteration. Fixtures with costly reads use fewer, so that each benchmark stays under
/// 1 ms in the CodSpeed simulation.
fn accesses(name: &str) -> usize {
    match name {
        "map" | "scalar_fn_binary" | "piecewise_variable" => 32,
        "scalar_fn_nested" => 16,
        "list" | "listview" | "fixed_size_list" => 64,
        "scalar_fn_validity" => 128,
        _ => ACCESSES,
    }
}

fn indices(count: usize) -> Vec<usize> {
    let mut seed = 42u64;
    (0..count)
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
    "bool",
    "map",
    "scalar_fn_binary",
    "scalar_fn_nested",
    "patched",
    "piecewise_variable",
    "piecewise_constant",
    "varbin",
    "list",
    "listview",
    "fixed_size_list",
    "interleave",
    "scalar_fn_validity",
];

const SCALAR_FN_ARRAYS: &[&str] = &["scalar_fn_binary", "scalar_fn_nested"];
const PIECEWISE_ARRAYS: &[&str] = &["piecewise_variable", "piecewise_constant"];

fn array(name: &str) -> ArrayRef {
    match name {
        "primitive" => nullable_primitive(LEN),
        "map" => nullable_map(),
        "scalar_fn_binary" => scalar_fn(false),
        "scalar_fn_nested" => scalar_fn(true),
        "patched" => patched(),
        "piecewise_variable" => piecewise(true),
        "piecewise_constant" => piecewise(false),
        "list" => nullable_list(),
        "listview" => nullable_listview(),
        "fixed_size_list" => nullable_fixed_size_list(),
        "scalar_fn_validity" => kleene_and_validity(),
        "interleave" => nullable_interleave(),
        "dict" => dict_over_nullable_values(),
        "struct" => nullable_struct(),
        "bool" => {
            BoolArray::from_iter((0..LEN).map(|i| (i % 7 != 0).then_some(i % 2 == 0))).into_array()
        }
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
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = indices(accesses(name));
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
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = indices(accesses(name));
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

#[divan::bench]
fn patched_same_lane_once(bencher: Bencher) {
    let array = patched();
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = same_lane_indices();
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

#[divan::bench]
fn patched_same_lane_repeated(bencher: Bencher) {
    let array = patched();
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = same_lane_indices();
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

#[divan::bench]
fn patched_chunked_indices_same_lane_repeated(bencher: Bencher) {
    let array = patched_with_chunked_indices();
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = same_lane_indices();
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

#[divan::bench(args = PIECEWISE_ARRAYS)]
fn piecewise_same_piece_once(bencher: Bencher, name: &str) {
    let array = array(name);
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = same_piece_indices(name == "piecewise_variable");
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

#[divan::bench(args = PIECEWISE_ARRAYS)]
fn piecewise_same_piece_repeated(bencher: Bencher, name: &str) {
    let array = array(name);
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    let indices = same_piece_indices(name == "piecewise_variable");
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
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for _ in 0..accesses(name) {
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
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), array.repeated_probe()))
        .bench_refs(|(ctx, probe)| {
            for _ in 0..accesses(name) {
                divan::black_box(
                    probe
                        .execute_scalar(LEN / 2, ctx)
                        .vortex_expect("in-bounds read"),
                );
            }
        });
}
