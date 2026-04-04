// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const LEN: usize = 1_000_000;

#[derive(Clone, Copy, Debug)]
enum SequenceShape {
    Ascending,
    Descending,
}

impl SequenceShape {
    fn build(self) -> vortex_sequence::SequenceArray {
        match self {
            Self::Ascending => Sequence::try_new_typed(10i64, 3, NonNullable, LEN).unwrap(),
            Self::Descending => {
                Sequence::try_new_typed(3_000_000i64, -3, NonNullable, LEN).unwrap()
            }
        }
    }

    fn midpoint_value(self) -> i64 {
        match self {
            Self::Ascending => 10 + 3 * (LEN / 2) as i64,
            Self::Descending => 3_000_000 - 3 * (LEN / 2) as i64,
        }
    }
}

impl fmt::Display for SequenceShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ascending => write!(f, "ascending"),
            Self::Descending => write!(f, "descending"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AffineExpr {
    SeqPlusConst,
    ConstPlusSeq,
    SeqMinusConst,
    ConstMinusSeq,
    SeqTimesConst,
    ConstTimesSeq,
}

impl AffineExpr {
    fn constant(self) -> i64 {
        match self {
            Self::SeqPlusConst | Self::ConstPlusSeq => 17,
            Self::SeqMinusConst => 11,
            Self::ConstMinusSeq => 10_000_000,
            Self::SeqTimesConst | Self::ConstTimesSeq => 2,
        }
    }

    fn operator(self) -> Operator {
        match self {
            Self::SeqPlusConst | Self::ConstPlusSeq => Operator::Add,
            Self::SeqMinusConst | Self::ConstMinusSeq => Operator::Sub,
            Self::SeqTimesConst | Self::ConstTimesSeq => Operator::Mul,
        }
    }

    fn midpoint_value(self, shape: SequenceShape) -> i64 {
        let midpoint = shape.midpoint_value();
        match self {
            Self::SeqPlusConst | Self::ConstPlusSeq => midpoint + self.constant(),
            Self::SeqMinusConst => midpoint - self.constant(),
            Self::ConstMinusSeq => self.constant() - midpoint,
            Self::SeqTimesConst | Self::ConstTimesSeq => midpoint * self.constant(),
        }
    }
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeqPlusConst => write!(f, "seq_plus_const"),
            Self::ConstPlusSeq => write!(f, "const_plus_seq"),
            Self::SeqMinusConst => write!(f, "seq_minus_const"),
            Self::ConstMinusSeq => write!(f, "const_minus_seq"),
            Self::SeqTimesConst => write!(f, "seq_times_const"),
            Self::ConstTimesSeq => write!(f, "const_times_seq"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchArgs {
    shape: SequenceShape,
    affine: AffineExpr,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.shape, self.affine)
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        shape: SequenceShape::Ascending,
        affine: AffineExpr::SeqPlusConst,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        affine: AffineExpr::ConstPlusSeq,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        affine: AffineExpr::SeqMinusConst,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        affine: AffineExpr::ConstMinusSeq,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        affine: AffineExpr::SeqTimesConst,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        affine: AffineExpr::ConstTimesSeq,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        affine: AffineExpr::SeqPlusConst,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        affine: AffineExpr::ConstPlusSeq,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        affine: AffineExpr::SeqMinusConst,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        affine: AffineExpr::ConstMinusSeq,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        affine: AffineExpr::SeqTimesConst,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        affine: AffineExpr::ConstTimesSeq,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn affine_compare_to_constant(bencher: Bencher, args: BenchArgs) {
    let sequence = args.shape.build().into_array();
    let affine_constant = ConstantArray::new(args.affine.constant(), LEN).into_array();
    let compare_constant =
        ConstantArray::new(args.affine.midpoint_value(args.shape), LEN).into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| {
            (
                sequence.clone(),
                affine_constant.clone(),
                compare_constant.clone(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(sequence, affine_constant, compare_constant, ctx)| {
            apply_affine(sequence.clone(), affine_constant.clone(), args.affine)
                .binary(compare_constant.clone(), Operator::Eq)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap();
        });
}

#[divan::bench(args = BENCH_ARGS)]
fn affine_transform(bencher: Bencher, args: BenchArgs) {
    let sequence = args.shape.build().into_array();
    let affine_constant = ConstantArray::new(args.affine.constant(), LEN).into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| {
            (
                sequence.clone(),
                affine_constant.clone(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(sequence, affine_constant, ctx)| {
            apply_affine(sequence.clone(), affine_constant.clone(), args.affine)
                .execute::<Canonical>(ctx)
                .unwrap();
        });
}

fn apply_affine(
    sequence: vortex_array::ArrayRef,
    affine_constant: vortex_array::ArrayRef,
    affine: AffineExpr,
) -> vortex_array::ArrayRef {
    match affine {
        AffineExpr::SeqPlusConst | AffineExpr::SeqMinusConst | AffineExpr::SeqTimesConst => {
            sequence.binary(affine_constant, affine.operator()).unwrap()
        }
        AffineExpr::ConstPlusSeq | AffineExpr::ConstMinusSeq | AffineExpr::ConstTimesSeq => {
            affine_constant.binary(sequence, affine.operator()).unwrap()
        }
    }
}
