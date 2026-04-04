// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation, clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_runend::RunEnd;

fn main() {
    divan::main();
}

const LEN: usize = 1_048_576;

#[derive(Clone, Copy, Debug)]
enum OutputKind {
    Utf8,
    Binary,
}

impl fmt::Display for OutputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8 => write!(f, "utf8"),
            Self::Binary => write!(f, "binary"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchArgs {
    run_length: usize,
    output_kind: OutputKind,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_runs_{}", self.output_kind, self.run_length)
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        run_length: 16,
        output_kind: OutputKind::Utf8,
    },
    BenchArgs {
        run_length: 256,
        output_kind: OutputKind::Utf8,
    },
    BenchArgs {
        run_length: 4096,
        output_kind: OutputKind::Utf8,
    },
    BenchArgs {
        run_length: 16,
        output_kind: OutputKind::Binary,
    },
    BenchArgs {
        run_length: 256,
        output_kind: OutputKind::Binary,
    },
    BenchArgs {
        run_length: 4096,
        output_kind: OutputKind::Binary,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn zip_constants(bencher: Bencher, args: BenchArgs) {
    let mask = bool_run_end_fixture(args.run_length);
    let (if_true, if_false) = constants(args.output_kind);

    bencher
        .with_inputs(|| {
            (
                mask.clone(),
                if_true.clone(),
                if_false.clone(),
                LEGACY_SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(mask, if_true, if_false, ctx)| {
            let result = mask
                .zip(if_true.clone(), if_false.clone())
                .unwrap()
                .execute::<ArrayRef>(ctx)
                .unwrap();
            divan::black_box(result);
        });
}

fn bool_run_end_fixture(run_length: usize) -> ArrayRef {
    let run_count = LEN.div_ceil(run_length);
    let ends = (0..run_count)
        .map(|run_idx| ((run_idx + 1) * run_length).min(LEN) as u32)
        .collect::<Buffer<_>>()
        .into_array();
    let values =
        BoolArray::from_iter((0..run_count).map(|run_idx| run_idx.is_multiple_of(2))).into_array();

    RunEnd::new(ends, values).into_array()
}

fn constants(output_kind: OutputKind) -> (ArrayRef, ArrayRef) {
    match output_kind {
        OutputKind::Utf8 => (
            ConstantArray::new(
                Scalar::utf8(
                    "runend branch with a long utf8 payload",
                    Nullability::NonNullable,
                ),
                LEN,
            )
            .into_array(),
            ConstantArray::new(
                Scalar::utf8(
                    "runend branch with a different utf8 payload",
                    Nullability::NonNullable,
                ),
                LEN,
            )
            .into_array(),
        ),
        OutputKind::Binary => (
            ConstantArray::new(
                Scalar::binary(vec![0xAA; 48], Nullability::NonNullable),
                LEN,
            )
            .into_array(),
            ConstantArray::new(
                Scalar::binary(vec![0x55; 64], Nullability::NonNullable),
                LEN,
            )
            .into_array(),
        ),
    }
}
