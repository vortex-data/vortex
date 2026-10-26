// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public PCO operations with fixed workloads shared by every implementation.
//! Compression/index fixtures and warm-cache preparation are outside the measured operation.
//! Retained batches have PROBES requests; fresh-state batches have FIRST_PROBES requests.
//! Divide by the corresponding count for per-probe time. Fresh state excludes owner/index opening.

use std::hint::black_box;
use std::sync::LazyLock;

use divan::Bencher;
use divan::main as run_benchmarks;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_pco::Pco;
use vortex_session::VortexSession;

const ROWS: usize = 4_194_304;
const PROBES: usize = 1_048_576;
const FIRST_PROBES: usize = 256;
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<KernelSession>()
        .with::<MemorySession>();
    session.arrays().register(Pco);
    session
});

fn main() {
    run_benchmarks();
}

fn input(nullable: bool) -> PrimitiveArray {
    PrimitiveArray::new(
        (0..ROWS)
            .map(|i| (i as i64).wrapping_mul(17) ^ ((i / 1024) as i64))
            .collect::<Vec<_>>(),
        if nullable {
            Validity::from_iter((0..ROWS).map(|i| i % 7 != 0))
        } else {
            Validity::NonNullable
        },
    )
}

fn compressed(nullable: bool) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    Pco::from_primitive(input(nullable).as_view(), 3, 1024, &mut ctx)
        .vortex_expect("valid benchmark fixture")
        .into_array()
}

#[divan::bench(args = [false, true])]
fn write(bencher: Bencher, nullable: bool) {
    let input = input(nullable);
    let mut check = SESSION.create_execution_ctx();
    let written =
        Pco::from_primitive(input.as_view(), 3, 1024, &mut check).vortex_expect("write oracle");
    assert_arrays_eq!(written.into_array(), input.clone(), &mut check);
    drop(check);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            Pco::from_primitive(input.as_view(), 3, 1024, ctx).vortex_expect("compression")
        });
}

#[divan::bench(args = [false, true])]
fn full(bencher: Bencher, nullable: bool) {
    let source = compressed(nullable);
    let mut check = SESSION.create_execution_ctx();
    assert_arrays_eq!(source, input(nullable), &mut check);
    drop(check);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            source
                .clone()
                .execute::<PrimitiveArray>(ctx)
                .vortex_expect("full decode")
        });
}

const SELECTIONS: &[(&str, bool)] = &[
    ("contiguous", false),
    ("contiguous", true),
    ("mixed", false),
    ("mixed", true),
    ("narrow_repeated", false),
    ("narrow_repeated", true),
    ("sparse", false),
    ("sparse", true),
    ("broad", false),
    ("broad", true),
    ("repeated", false),
    ("repeated", true),
    ("near_full", false),
    ("near_full", true),
    ("ranges", false),
    ("ranges", true),
];

#[divan::bench(args = SELECTIONS)]
fn selection(bencher: Bencher, (kind, nullable): (&str, bool)) {
    let source = compressed(nullable);
    let source = if kind == "narrow_repeated" {
        source.slice(ROWS - 128..ROWS).vortex_expect("narrow view")
    } else {
        source
    };

    let indices = match kind {
        "contiguous" => PrimitiveArray::from_iter(1..ROWS as u64 - 1).into_array(),
        "mixed" => PrimitiveArray::from_iter(
            (0..524288u64).map(|i| if i < 2 { i } else { (i * 32749) % ROWS as u64 }),
        )
        .into_array(),
        "narrow_repeated" => {
            PrimitiveArray::from_iter((0..4_194_304u64).map(|i| (i % 2) * 127)).into_array()
        }
        "sparse" => {
            PrimitiveArray::from_iter((0..128u64).map(|i| (i * 32749) % ROWS as u64)).into_array()
        }
        "broad" => PrimitiveArray::from_iter((0..524288u64).map(|i| (i * 32749) % ROWS as u64))
            .into_array(),
        "repeated" => {
            PrimitiveArray::from_iter((0..524288u64).map(|i| (i % 16) * 127)).into_array()
        }
        "ranges" => PiecewiseSequenceArray::try_new(
            PrimitiveArray::from_iter([0u64, 100000, 200000, 300000]).into_array(),
            PrimitiveArray::from_iter([512u64; 4]).into_array(),
            PrimitiveArray::from_iter([1u64; 4]).into_array(),
            2048,
        )
        .vortex_expect("range indices")
        .into_array(),
        _ => PrimitiveArray::from_iter([0u64]).into_array(),
    };
    let mut check = SESSION.create_execution_ctx();
    let reference = input(nullable).into_array();
    let reference = if kind == "narrow_repeated" {
        reference
            .slice(ROWS - 128..ROWS)
            .vortex_expect("primitive narrow view")
    } else {
        reference
    };
    let actual = if kind == "near_full" {
        source.slice(1..ROWS - 1)
    } else {
        source.take(indices.clone())
    }
    .vortex_expect("selection oracle");
    let expected = if kind == "near_full" {
        reference.slice(1..ROWS - 1)
    } else {
        reference.take(indices.clone())
    }
    .vortex_expect("primitive oracle");
    assert_arrays_eq!(actual, expected, &mut check);
    drop(reference);
    drop(check);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            let result = if kind == "near_full" {
                source.slice(1..ROWS - 1).vortex_expect("slice")
            } else {
                source.take(indices.clone()).vortex_expect("take")
            };
            result
                .execute::<PrimitiveArray>(ctx)
                .vortex_expect("selection")
        });
}

#[divan::bench(args = [false, true])]
fn retained_probes(bencher: Bencher, nullable: bool) {
    let source = compressed(nullable);
    let reference = input(nullable).into_array();
    let mut check = SESSION.create_execution_ctx();
    let mut checked_probe = source.repeated_probe();
    for i in 0..256 {
        assert_eq!(
            checked_probe
                .execute_scalar(8192 + i, &mut check)
                .vortex_expect("probe oracle"),
            reference
                .execute_scalar(8192 + i, &mut check)
                .vortex_expect("primitive probe")
        );
    }

    drop(checked_probe);
    drop(reference);
    drop(check);
    bencher
        .with_inputs(|| {
            let mut ctx = SESSION.create_execution_ctx();
            let mut probe = source.repeated_probe();
            for i in 0..256 {
                black_box(
                    probe
                        .execute_scalar(8192 + i, &mut ctx)
                        .vortex_expect("warm page"),
                );
            }
            (ctx, probe)
        })
        .bench_refs(|(ctx, probe)| {
            for i in 0..PROBES {
                black_box(
                    probe
                        .execute_scalar(8192 + i % 256, ctx)
                        .vortex_expect("retained probe"),
                );
            }
        });
}

// Each request starts a new probe. Missing and defined rows are separate cases.
#[divan::bench(args = [false, true])]
fn first_probes(bencher: Bencher, missing: bool) {
    let source = compressed(true);
    let index = if missing { 8190 } else { 8191 };
    let mut check = SESSION.create_execution_ctx();
    let actual = source
        .probe()
        .execute_scalar(index, &mut check)
        .vortex_expect("fresh probe oracle");
    assert_eq!(
        actual,
        input(true)
            .into_array()
            .execute_scalar(index, &mut check)
            .vortex_expect("primitive oracle")
    );
    assert_eq!(actual.is_null(), missing);
    drop(actual);

    drop(check);
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for _ in 0..FIRST_PROBES {
                black_box(
                    source
                        .probe()
                        .execute_scalar(index, ctx)
                        .vortex_expect("first probe"),
                );
            }
        });
}

// Buffers/children are already materialized: this measures metadata/index preparation, not IO.
#[divan::bench(args = [false, true])]
fn metadata_open_batch(bencher: Bencher, nullable: bool) {
    const OPENS: usize = 256;
    let source = compressed(nullable);
    let wire = SESSION
        .array_serialize(&source)
        .vortex_expect("serialize metadata fixture")
        .vortex_expect("PCO is serializable");
    let buffers = wire
        .buffers
        .into_iter()
        .map(BufferHandle::new_host)
        .collect::<Vec<_>>();
    let open = || {
        ArrayPlugin::deserialize(
            &Pco,
            ArrayDeserialization::new(
                wire.serialized_id,
                source.dtype(),
                source.len(),
                &wire.metadata,
                &buffers,
                &wire.children,
            ),
            &SESSION,
        )
        .vortex_expect("public metadata open")
    };
    let mut check = SESSION.create_execution_ctx();
    assert_arrays_eq!(open(), input(nullable), &mut check);
    drop(check);
    bencher.bench_local(|| {
        for _ in 0..OPENS {
            black_box(open());
        }
    });
}
