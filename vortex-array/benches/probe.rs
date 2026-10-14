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
use vortex_array::arrays::DictArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::optimizer::kernels::KernelSession;
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
    "list",
    "interleave",
];

fn array(name: &str) -> ArrayRef {
    match name {
        "primitive" => nullable_primitive(LEN),
        "list" => nullable_list(),
        "interleave" => nullable_interleave(),
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
