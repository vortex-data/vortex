// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::fmt;

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
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
    Constant,
}

impl SequenceShape {
    fn build(self) -> vortex_sequence::SequenceArray {
        match self {
            Self::Ascending => Sequence::try_new_typed(10i64, 3, NonNullable, LEN).unwrap(),
            Self::Descending => {
                Sequence::try_new_typed(3_000_000i64, -3, NonNullable, LEN).unwrap()
            }
            Self::Constant => Sequence::try_new_typed(42i64, 0, NonNullable, LEN).unwrap(),
        }
    }
}

impl fmt::Display for SequenceShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ascending => write!(f, "ascending"),
            Self::Descending => write!(f, "descending"),
            Self::Constant => write!(f, "constant"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ConstantCase {
    AtStart,
    AtMiddle,
    AtEnd,
    BelowRange,
    AboveRange,
}

impl ConstantCase {
    fn value(self, shape: SequenceShape) -> i64 {
        match (shape, self) {
            (SequenceShape::Ascending, Self::AtStart) => 10,
            (SequenceShape::Ascending, Self::AtMiddle) => 10 + 3 * (LEN / 2) as i64,
            (SequenceShape::Ascending, Self::AtEnd) => 10 + 3 * (LEN - 1) as i64,
            (SequenceShape::Ascending, Self::BelowRange) => 9,
            (SequenceShape::Ascending, Self::AboveRange) => 10 + 3 * LEN as i64,
            (SequenceShape::Descending, Self::AtStart) => 3_000_000,
            (SequenceShape::Descending, Self::AtMiddle) => 3_000_000 - 3 * (LEN / 2) as i64,
            (SequenceShape::Descending, Self::AtEnd) => 3_000_000 - 3 * (LEN - 1) as i64,
            (SequenceShape::Descending, Self::BelowRange) => 3_000_000 - 3 * LEN as i64,
            (SequenceShape::Descending, Self::AboveRange) => 3_000_001,
            (SequenceShape::Constant, Self::AtStart | Self::AtMiddle | Self::AtEnd) => 42,
            (SequenceShape::Constant, Self::BelowRange) => 41,
            (SequenceShape::Constant, Self::AboveRange) => 43,
        }
    }
}

impl fmt::Display for ConstantCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtStart => write!(f, "start"),
            Self::AtMiddle => write!(f, "middle"),
            Self::AtEnd => write!(f, "end"),
            Self::BelowRange => write!(f, "below_range"),
            Self::AboveRange => write!(f, "above_range"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchArgs {
    shape: SequenceShape,
    operator: Operator,
    constant_case: ConstantCase,
}

impl fmt::Display for BenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}_{}", self.shape, self.operator, self.constant_case)
    }
}

#[derive(Clone, Copy, Debug)]
struct EqControlArgs {
    shape: SequenceShape,
    constant_case: ConstantCase,
}

impl fmt::Display for EqControlArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_eq_{}", self.shape, self.constant_case)
    }
}

#[derive(Clone, Copy, Debug)]
struct FallbackControlArgs {
    shape: SequenceShape,
    operator: Operator,
    constant_case: ConstantCase,
}

impl fmt::Display for FallbackControlArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}_fallback",
            self.shape, self.operator, self.constant_case
        )
    }
}

const BENCH_ARGS: &[BenchArgs] = &[
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lte,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gt,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lte,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lte,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gt,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gt,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lte,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lte,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lte,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gt,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gt,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gt,
        constant_case: ConstantCase::AboveRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtStart,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtMiddle,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtEnd,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::BelowRange,
    },
    BenchArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::AboveRange,
    },
];

const EQ_CONTROL_ARGS: &[EqControlArgs] = &[
    EqControlArgs {
        shape: SequenceShape::Ascending,
        constant_case: ConstantCase::AtStart,
    },
    EqControlArgs {
        shape: SequenceShape::Ascending,
        constant_case: ConstantCase::AtMiddle,
    },
    EqControlArgs {
        shape: SequenceShape::Ascending,
        constant_case: ConstantCase::AtEnd,
    },
    EqControlArgs {
        shape: SequenceShape::Ascending,
        constant_case: ConstantCase::BelowRange,
    },
    EqControlArgs {
        shape: SequenceShape::Ascending,
        constant_case: ConstantCase::AboveRange,
    },
    EqControlArgs {
        shape: SequenceShape::Descending,
        constant_case: ConstantCase::AtStart,
    },
    EqControlArgs {
        shape: SequenceShape::Descending,
        constant_case: ConstantCase::AtMiddle,
    },
    EqControlArgs {
        shape: SequenceShape::Descending,
        constant_case: ConstantCase::AtEnd,
    },
    EqControlArgs {
        shape: SequenceShape::Descending,
        constant_case: ConstantCase::BelowRange,
    },
    EqControlArgs {
        shape: SequenceShape::Descending,
        constant_case: ConstantCase::AboveRange,
    },
    EqControlArgs {
        shape: SequenceShape::Constant,
        constant_case: ConstantCase::AtStart,
    },
    EqControlArgs {
        shape: SequenceShape::Constant,
        constant_case: ConstantCase::AtMiddle,
    },
    EqControlArgs {
        shape: SequenceShape::Constant,
        constant_case: ConstantCase::AtEnd,
    },
    EqControlArgs {
        shape: SequenceShape::Constant,
        constant_case: ConstantCase::BelowRange,
    },
    EqControlArgs {
        shape: SequenceShape::Constant,
        constant_case: ConstantCase::AboveRange,
    },
];

const FALLBACK_CONTROL_ARGS: &[FallbackControlArgs] = &[
    FallbackControlArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtMiddle,
    },
    FallbackControlArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AboveRange,
    },
    FallbackControlArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtMiddle,
    },
    FallbackControlArgs {
        shape: SequenceShape::Ascending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AboveRange,
    },
    FallbackControlArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtMiddle,
    },
    FallbackControlArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Lt,
        constant_case: ConstantCase::AboveRange,
    },
    FallbackControlArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtMiddle,
    },
    FallbackControlArgs {
        shape: SequenceShape::Descending,
        operator: Operator::Gte,
        constant_case: ConstantCase::AboveRange,
    },
    FallbackControlArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::AtMiddle,
    },
    FallbackControlArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Lt,
        constant_case: ConstantCase::AboveRange,
    },
    FallbackControlArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::AtMiddle,
    },
    FallbackControlArgs {
        shape: SequenceShape::Constant,
        operator: Operator::Gte,
        constant_case: ConstantCase::AboveRange,
    },
];

#[divan::bench(args = BENCH_ARGS)]
fn compare_to_constant(bencher: Bencher, args: BenchArgs) {
    let sequence = args.shape.build();
    let rhs = ConstantArray::new(args.constant_case.value(args.shape), LEN).into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| {
            (
                sequence.clone().into_array(),
                rhs.clone(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(lhs, rhs, ctx)| {
            lhs.clone()
                .binary(rhs.clone(), args.operator)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = EQ_CONTROL_ARGS)]
fn compare_eq_control(bencher: Bencher, args: EqControlArgs) {
    let sequence = args.shape.build();
    let rhs = ConstantArray::new(args.constant_case.value(args.shape), LEN).into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| {
            (
                sequence.clone().into_array(),
                rhs.clone(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(lhs, rhs, ctx)| {
            lhs.clone()
                .binary(rhs.clone(), Operator::Eq)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = FALLBACK_CONTROL_ARGS)]
fn compare_non_constant_control(bencher: Bencher, args: FallbackControlArgs) {
    let sequence = args.shape.build();
    let rhs = PrimitiveArray::from_iter((0..LEN).map(|_| args.constant_case.value(args.shape)))
        .into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| {
            (
                sequence.clone().into_array(),
                rhs.clone(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(lhs, rhs, ctx)| {
            lhs.clone()
                .binary(rhs.clone(), args.operator)
                .unwrap()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}
