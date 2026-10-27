// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute on integer `Widen` arrays, against the canonical array with the same values.
//!
//! Each case runs one operation on the inputs of [`Input`], which hold the same logical `i64`
//! values. `slice` and `add_canonical` omit `canonical`. A kernel for `Widen` changes only the
//! `widen` case. The `canonical` case measures the canonical path, and the `decoded` case
//! measures the fallback that decodes the `Widen` first. So the `canonical` and `decoded` cases
//! must stay flat when a kernel for `Widen` lands.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max;
use vortex_array::aggregate_fn::fns::sum;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);

    // The first execution initializes process-wide state. Without this call, that work lands in the
    // single measured iteration of the first benchmark under the CodSpeed simulation.
    let values = Values::random(0, LEN);
    Input::Decoded.prepare(
        values.array(Input::Widen),
        &mut SESSION.create_execution_ctx(),
    );

    divan::main();
}

// The session variables are created here, so that no timed case pays for their default
// constructors. No case serializes, so the session needs no `ArraySession`.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>()
});

// The canonical input holds 256 KiB. The inputs are sized so that every case does a few
// microseconds of real work above the CodSpeed harness floor, while the slowest case stays far
// under 1 ms.
const LEN: usize = 32_768;

/// The number of values of the dictionary in `dict_take`, so that a `u8` code can index each one.
const DICT_VALUES_LEN: usize = 256;

const I64: DType = DType::Primitive(PType::I64, Nullability::NonNullable);
const F64: DType = DType::Primitive(PType::F64, Nullability::NonNullable);

const INCLUSIVE: BetweenOptions = BetweenOptions {
    lower_strict: StrictComparison::NonStrict,
    upper_strict: StrictComparison::NonStrict,
};

#[derive(Clone, Copy, Debug)]
enum Input {
    /// The canonical `i64` array with the same values.
    Canonical,
    /// A `Widen` with logical dtype `i64` over an `i16` child.
    Widen,
    /// The `Widen`, executed to a `PrimitiveArray` inside the timed region, then the operation.
    Decoded,
}

const INPUTS: &[Input] = &[Input::Canonical, Input::Widen, Input::Decoded];
const WIDEN_INPUTS: &[Input] = &[Input::Widen, Input::Decoded];

impl Input {
    /// Returns `array` as the operation receives it. For `Decoded`, this decodes the `Widen`, so
    /// it must run inside the timed region.
    fn prepare(self, array: ArrayRef, ctx: &mut ExecutionCtx) -> ArrayRef {
        match self {
            Self::Canonical | Self::Widen => array,
            Self::Decoded => array
                .execute::<PrimitiveArray>(ctx)
                .vortex_expect("a Widen of i64 always executes to a PrimitiveArray")
                .into_array(),
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Canonical => "canonical",
            Self::Widen => "widen",
            Self::Decoded => "decoded",
        })
    }
}

/// The buffers of one input, shared by every iteration.
///
/// `sum` caches its result as a statistic on the array that it reads, and later kernels read
/// statistics of their operands, such as the extrema. A reused array would hide that work. So each
/// iteration builds new arrays from these buffers, which shares the memory but not the statistics,
/// and no case clones a built `ArrayRef`.
struct Values {
    narrow: Buffer<i16>,
    wide: Buffer<i64>,
    /// The validity of a nullable input, or `None` for a non-nullable one.
    validity: Option<BitBuffer>,
}

impl Values {
    /// `len` non-nullable values, uniform in `-10_000..=10_000`.
    fn random(seed: u64, len: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let narrow: Buffer<i16> = (0..len)
            .map(|_| rng.random_range(-10_000..=10_000))
            .collect();
        let wide = narrow.iter().copied().map(i64::from).collect();

        Self {
            narrow,
            wide,
            validity: None,
        }
    }

    /// Makes the values nullable, with `validity` as their validity.
    fn with_validity(self, validity: BitBuffer) -> Self {
        Self {
            validity: Some(validity),
            ..self
        }
    }

    /// Builds a new array for `input`. `Decoded` gets the `Widen`, which [`Input::prepare`]
    /// decodes.
    fn array(&self, input: Input) -> ArrayRef {
        let validity = self
            .validity
            .clone()
            .map_or(Validity::NonNullable, Validity::from);

        match input {
            Input::Canonical => PrimitiveArray::new(self.wide.clone(), validity).into_array(),
            Input::Widen | Input::Decoded => {
                let dtype = DType::Primitive(PType::I64, validity.nullability());
                let values = PrimitiveArray::new(self.narrow.clone(), validity).into_array();
                WidenArray::try_new(values, dtype)
                    .vortex_expect("i16 values with the nullability of the dtype can widen to i64")
                    .into_array()
            }
        }
    }
}

/// [`LEN`] random bits, each set with probability `density`.
fn random_bits(seed: u64, density: f64) -> BitBuffer {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..LEN).map(|_| rng.random_bool(density)).collect()
}

fn constant(value: i64) -> ArrayRef {
    ConstantArray::new(value, LEN).into_array()
}

/// Runs `operation` on operands that `setup` builds anew for each iteration.
fn bench_case<T, O>(
    bencher: Bencher,
    setup: impl Fn() -> T + Sync,
    operation: impl Fn(T, &mut ExecutionCtx) -> O + Sync,
) {
    bencher
        .with_inputs(|| (setup(), SESSION.create_execution_ctx()))
        .bench_values(|(operands, mut ctx)| operation(operands, &mut ctx));
}

fn execute_canonical(array: VortexResult<ArrayRef>, ctx: &mut ExecutionCtx) -> Canonical {
    array
        .vortex_expect("every operation is valid for the benchmark inputs")
        .execute::<Canonical>(ctx)
        .vortex_expect("every operation executes to canonical for the benchmark inputs")
}

#[divan::bench(args = INPUTS)]
fn compare_in_range(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || (values.array(input), constant(0)),
        |(array, zero), ctx| {
            let array = input.prepare(array, ctx);
            execute_canonical(array.binary(zero, Operator::Lt), ctx)
        },
    );
}

#[divan::bench(args = INPUTS)]
fn between_in_range(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || (values.array(input), constant(-1_000), constant(1_000)),
        |(array, lower, upper), ctx| {
            let array = input.prepare(array, ctx);
            execute_canonical(array.between(lower, upper, INCLUSIVE), ctx)
        },
    );
}

#[divan::bench(args = INPUTS)]
fn filter(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    let selection = random_bits(2, 0.5);
    bench_case(
        bencher,
        || (values.array(input), Mask::from_buffer(selection.clone())),
        |(array, mask), ctx| execute_canonical(input.prepare(array, ctx).filter(mask), ctx),
    );
}

#[divan::bench(args = INPUTS)]
fn take(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    let mut rng = StdRng::seed_from_u64(3);
    let len = u32::try_from(LEN).vortex_expect("LEN fits u32");
    let take_indices: Buffer<u32> = (0..LEN / 4).map(|_| rng.random_range(0..len)).collect();
    bench_case(
        bencher,
        || (values.array(input), take_indices.clone().into_array()),
        |(array, indices), ctx| execute_canonical(input.prepare(array, ctx).take(indices), ctx),
    );
}

/// A dictionary whose codes far outnumber its values.
///
/// Under a real `Dict`, the `Widen` take kernel gathers every code from the narrow values, and
/// execution then widens the gathered rows. The `decoded` input widens only the dictionary values.
#[divan::bench(args = INPUTS)]
fn dict_take(bencher: Bencher, input: Input) {
    let values = Values::random(0, DICT_VALUES_LEN);
    let mut rng = StdRng::seed_from_u64(3);
    let codes: Buffer<u8> = (0..LEN)
        .map(|_| {
            u8::try_from(rng.random_range(0..DICT_VALUES_LEN))
                .vortex_expect("DICT_VALUES_LEN keeps every code within u8")
        })
        .collect();
    bench_case(
        bencher,
        || (values.array(input), codes.clone().into_array()),
        |(values, codes), ctx| {
            let values = input.prepare(values, ctx);
            execute_canonical(
                DictArray::try_new(codes, values).map(IntoArray::into_array),
                ctx,
            )
        },
    );
}

/// Omits the `canonical` input, because a slice of a canonical array copies nothing.
#[divan::bench(args = WIDEN_INPUTS)]
fn slice(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.array(input),
        |array, ctx| execute_canonical(input.prepare(array, ctx).slice(LEN / 4..3 * LEN / 4), ctx),
    );
}

#[divan::bench(args = INPUTS)]
fn sum(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.array(input),
        |array, ctx| {
            sum::sum(&input.prepare(array, ctx), ctx)
                .vortex_expect("sum is valid for every benchmark input")
        },
    );
}

#[divan::bench(args = INPUTS)]
fn min_max(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.array(input),
        |array, ctx| {
            let array = input.prepare(array, ctx);
            min_max::min_max(&array, ctx, NumericalAggregateOpts::default())
                .vortex_expect("min_max is valid for every benchmark input")
        },
    );
}

#[divan::bench(args = INPUTS)]
fn add_constant(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || (values.array(input), constant(1)),
        |(array, one), ctx| {
            let array = input.prepare(array, ctx);
            execute_canonical(array.binary(one, Operator::Add), ctx)
        },
    );
}

#[divan::bench(args = INPUTS)]
fn add_widen(bencher: Bencher, input: Input) {
    let lhs = Values::random(0, LEN);
    let rhs = Values::random(1, LEN);
    bench_case(
        bencher,
        || (lhs.array(input), rhs.array(input)),
        |(lhs, rhs), ctx| {
            let lhs = input.prepare(lhs, ctx);
            let rhs = input.prepare(rhs, ctx);
            execute_canonical(lhs.binary(rhs, Operator::Add), ctx)
        },
    );
}

/// Adds a canonical `i64` array, which a `Widen` kernel cannot read at the child width.
#[divan::bench(args = WIDEN_INPUTS)]
fn add_canonical(bencher: Bencher, input: Input) {
    let lhs = Values::random(0, LEN);
    let rhs = Values::random(1, LEN);
    bench_case(
        bencher,
        || (lhs.array(input), rhs.array(Input::Canonical)),
        |(lhs, rhs), ctx| {
            let lhs = input.prepare(lhs, ctx);
            execute_canonical(lhs.binary(rhs, Operator::Add), ctx)
        },
    );
}

#[divan::bench(args = INPUTS)]
fn cast_f64(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.array(input),
        |array, ctx| execute_canonical(input.prepare(array, ctx).cast(F64), ctx),
    );
}

/// Fills the nulls with a value that fits `i32` but not the `i16` child.
#[divan::bench(args = INPUTS)]
fn fill_null_out_of_range(bencher: Bencher, input: Input) {
    let values = Values::random(0, LEN).with_validity(random_bits(4, 0.9));
    bench_case(
        bencher,
        || values.array(input),
        |array, ctx| execute_canonical(input.prepare(array, ctx).fill_null(100_000i64), ctx),
    );
}

#[divan::bench(args = INPUTS)]
fn zip(bencher: Bencher, input: Input) {
    let if_true = Values::random(0, LEN);
    let if_false = Values::random(1, LEN);
    let selection = random_bits(2, 0.5);
    bench_case(
        bencher,
        || {
            let mask = selection.clone().into_array();
            (mask, if_true.array(input), if_false.array(input))
        },
        |(mask, if_true, if_false), ctx| {
            let if_true = input.prepare(if_true, ctx);
            let if_false = input.prepare(if_false, ctx);
            execute_canonical(mask.zip(if_true, if_false), ctx)
        },
    );
}

/// Canonicalizes a chunked array of four chunks, whose chunks are `Widen` for the `widen` input.
#[divan::bench(args = INPUTS)]
fn chunked(bencher: Bencher, input: Input) {
    let chunks: Vec<Values> = (0..4).map(|seed| Values::random(seed, LEN / 4)).collect();
    bench_case(
        bencher,
        || {
            chunks
                .iter()
                .map(|chunk| chunk.array(input))
                .collect::<Vec<_>>()
        },
        |chunks, ctx| {
            let chunks: Vec<ArrayRef> = chunks
                .into_iter()
                .map(|chunk| input.prepare(chunk, ctx))
                .collect();
            execute_canonical(
                ChunkedArray::try_new(chunks, I64).map(IntoArray::into_array),
                ctx,
            )
        },
    );
}

/// Merges two inputs row by row.
#[divan::bench(args = INPUTS)]
fn interleave(bencher: Bencher, input: Input) {
    let first = Values::random(0, LEN / 2);
    let second = Values::random(1, LEN / 2);
    let array_indices: Buffer<u8> = (0..LEN / 2).flat_map(|_| [0, 1]).collect();
    let row_indices: Buffer<u32> = (0u32..).take(LEN / 2).flat_map(|row| [row, row]).collect();
    bench_case(
        bencher,
        || {
            let values = [first.array(input), second.array(input)];
            (
                values,
                array_indices.clone().into_array(),
                row_indices.clone().into_array(),
            )
        },
        |([first, second], array_indices, row_indices), ctx| {
            let values = vec![input.prepare(first, ctx), input.prepare(second, ctx)];
            let interleaved = InterleaveArray::try_new(values, array_indices, row_indices);
            execute_canonical(interleaved.map(IntoArray::into_array), ctx)
        },
    );
}
