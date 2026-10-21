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
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::patched::PatchedSlots;
use vortex_array::dtype::FieldNames;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::patches::Patches;
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
        indices.slice(0..middle).vortex_expect("first index chunk"),
        indices
            .slice(middle..indices.len())
            .vortex_expect("second index chunk"),
    ];
    let chunked = ChunkedArray::try_new(chunks, indices.dtype().clone())
        .vortex_expect("chunked patch indices")
        .into_array();
    let slots = PatchedSlots {
        inner: patched.inner().clone(),
        lane_offsets: patched.lane_offsets().clone(),
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
    (0..ACCESSES).map(|i| LEN / 2 + i % length).collect()
}

fn same_lane_indices() -> Vec<usize> {
    (0..ACCESSES).map(|i| (i % 32) * 32).collect()
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
    "patched",
    "piecewise_variable",
    "piecewise_constant",
];

const PIECEWISE_ARRAYS: &[&str] = &["piecewise_variable", "piecewise_constant"];

fn array(name: &str) -> ArrayRef {
    match name {
        "primitive" => nullable_primitive(LEN),
        "patched" => patched(),
        "piecewise_variable" => piecewise(true),
        "piecewise_constant" => piecewise(false),
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

#[divan::bench]
fn patched_same_lane_once(bencher: Bencher) {
    let array = patched();
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
