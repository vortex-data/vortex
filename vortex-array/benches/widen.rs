// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compute on integer `Widen` arrays.
//!
//! Each case runs one operation on a `Widen` with logical dtype `i64` over an `i16` child.
//! The `decode` case measures materialization separately from the operations that can use
//! the narrower child directly.

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
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::test_harness::mark_optimized;
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
    Values::random(0, LEN)
        .widen()
        .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
        .vortex_expect("a Widen of i64 always executes to a PrimitiveArray");

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

// The decoded values hold 256 KiB. The inputs are sized so that every case does a few
// microseconds of real work above the CodSpeed harness floor, while the slowest case stays far
// under 1 ms.
const LEN: usize = 32_768;

/// The number of values in `take_many_indices`, so that a `u8` index can address each one.
const FEW_VALUES_LEN: usize = 256;

const I64: DType = DType::Primitive(PType::I64, Nullability::NonNullable);
const F64: DType = DType::Primitive(PType::F64, Nullability::NonNullable);

const INCLUSIVE: BetweenOptions = BetweenOptions {
    lower_strict: StrictComparison::NonStrict,
    upper_strict: StrictComparison::NonStrict,
};

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

    fn validity(&self) -> Validity {
        self.validity
            .clone()
            .map_or(Validity::NonNullable, Validity::from)
    }

    /// Builds a new `Widen` of the values.
    fn widen(&self) -> ArrayRef {
        let validity = self.validity();
        let dtype = DType::Primitive(PType::I64, validity.nullability());
        let values = PrimitiveArray::new(self.narrow.clone(), validity).into_array();
        let array = WidenArray::try_new(values, dtype)
            .vortex_expect("i16 values with the nullability of the dtype can widen to i64")
            .into_array();
        mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");

        array
    }

    /// Builds a new canonical `i64` array of the values.
    fn canonical(&self) -> ArrayRef {
        let array = PrimitiveArray::new(self.wide.clone(), self.validity()).into_array();
        mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");

        array
    }
}

/// [`LEN`] random bits, each set with probability `density`.
fn random_bits(seed: u64, density: f64) -> BitBuffer {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..LEN).map(|_| rng.random_bool(density)).collect()
}

fn constant(value: i64) -> ArrayRef {
    let array = ConstantArray::new(value, LEN).into_array();
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");

    array
}

fn index_array<T: NativePType>(indices: &Buffer<T>) -> ArrayRef {
    let array = indices.clone().into_array();
    mark_optimized(&array, &SESSION).vortex_expect("optimize the benchmark input");

    array
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

#[divan::bench]
fn compare_in_range(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || (values.widen(), constant(0)),
        |(array, zero), ctx| execute_canonical(array.binary(zero, Operator::Lt), ctx),
    );
}

#[divan::bench]
fn between_in_range(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || (values.widen(), constant(-1_000), constant(1_000)),
        |(array, lower, upper), ctx| execute_canonical(array.between(lower, upper, INCLUSIVE), ctx),
    );
}

#[divan::bench]
fn filter(bencher: Bencher) {
    let values = Values::random(0, LEN);
    let selection = random_bits(2, 0.5);
    bench_case(
        bencher,
        || (values.widen(), Mask::from_buffer(selection.clone())),
        |(array, mask), ctx| execute_canonical(array.filter(mask), ctx),
    );
}

/// Gathers [`LEN`] / 4 rows from [`LEN`] values, so only the gathered rows are widened.
#[divan::bench]
fn take_few_indices(bencher: Bencher) {
    let values = Values::random(0, LEN);
    let mut rng = StdRng::seed_from_u64(3);
    let len = u32::try_from(LEN).vortex_expect("LEN fits u32");
    let indices: Buffer<u32> = (0..LEN / 4).map(|_| rng.random_range(0..len)).collect();
    bench_case(
        bencher,
        || (values.widen(), index_array(&indices)),
        |(array, indices), ctx| execute_canonical(array.take(indices), ctx),
    );
}

/// Gathers [`LEN`] rows from [`FEW_VALUES_LEN`] values, as a dictionary gathers its values by its
/// codes, so every gathered row is widened.
#[divan::bench]
fn take_many_indices(bencher: Bencher) {
    let values = Values::random(0, FEW_VALUES_LEN);
    let mut rng = StdRng::seed_from_u64(3);
    let indices: Buffer<u8> = (0..LEN)
        .map(|_| {
            u8::try_from(rng.random_range(0..FEW_VALUES_LEN))
                .vortex_expect("FEW_VALUES_LEN keeps every index within u8")
        })
        .collect();
    bench_case(
        bencher,
        || (values.widen(), index_array(&indices)),
        |(array, indices), ctx| execute_canonical(array.take(indices), ctx),
    );
}

/// Executes the `Widen` to canonical, which every operation without a kernel does first.
#[divan::bench]
fn decode(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.widen(),
        |array, ctx| {
            array
                .execute::<Canonical>(ctx)
                .vortex_expect("a Widen always executes to canonical")
        },
    );
}

#[divan::bench]
fn slice(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.widen(),
        |array, ctx| execute_canonical(array.slice(LEN / 4..3 * LEN / 4), ctx),
    );
}

#[divan::bench]
fn sum(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.widen(),
        |array, ctx| sum::sum(&array, ctx).vortex_expect("sum is valid for every benchmark input"),
    );
}

#[divan::bench]
fn min_max(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.widen(),
        |array, ctx| {
            min_max::min_max(&array, ctx, NumericalAggregateOpts::default())
                .vortex_expect("min_max is valid for every benchmark input")
        },
    );
}

#[divan::bench]
fn add_constant(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || (values.widen(), constant(1)),
        |(array, one), ctx| execute_canonical(array.binary(one, Operator::Add), ctx),
    );
}

#[divan::bench]
fn add_widen(bencher: Bencher) {
    let lhs = Values::random(0, LEN);
    let rhs = Values::random(1, LEN);
    bench_case(
        bencher,
        || (lhs.widen(), rhs.widen()),
        |(lhs, rhs), ctx| execute_canonical(lhs.binary(rhs, Operator::Add), ctx),
    );
}

/// Adds a canonical `i64` array, which a `Widen` kernel cannot read at the child width.
#[divan::bench]
fn add_canonical(bencher: Bencher) {
    let lhs = Values::random(0, LEN);
    let rhs = Values::random(1, LEN);
    bench_case(
        bencher,
        || (lhs.widen(), rhs.canonical()),
        |(lhs, rhs), ctx| execute_canonical(lhs.binary(rhs, Operator::Add), ctx),
    );
}

#[divan::bench]
fn cast_f64(bencher: Bencher) {
    let values = Values::random(0, LEN);
    bench_case(
        bencher,
        || values.widen(),
        |array, ctx| execute_canonical(array.cast(F64), ctx),
    );
}

/// Fills the nulls with a value that fits `i32` but not the `i16` child.
#[divan::bench]
fn fill_null_out_of_range(bencher: Bencher) {
    let values = Values::random(0, LEN).with_validity(random_bits(4, 0.9));
    bench_case(
        bencher,
        || values.widen(),
        |array, ctx| execute_canonical(array.fill_null(100_000i64), ctx),
    );
}

#[divan::bench]
fn zip(bencher: Bencher) {
    let if_true = Values::random(0, LEN);
    let if_false = Values::random(1, LEN);
    let selection = random_bits(2, 0.5);
    bench_case(
        bencher,
        || {
            let mask = selection.clone().into_array();
            mark_optimized(&mask, &SESSION).vortex_expect("optimize the benchmark input");

            (mask, if_true.widen(), if_false.widen())
        },
        |(mask, if_true, if_false), ctx| execute_canonical(mask.zip(if_true, if_false), ctx),
    );
}

/// Canonicalizes a chunked array of four `Widen` chunks.
#[divan::bench]
fn chunked(bencher: Bencher) {
    let chunks: Vec<Values> = (0..4).map(|seed| Values::random(seed, LEN / 4)).collect();
    bench_case(
        bencher,
        || chunks.iter().map(Values::widen).collect::<Vec<_>>(),
        |chunks, ctx| {
            execute_canonical(
                ChunkedArray::try_new(chunks, I64).map(IntoArray::into_array),
                ctx,
            )
        },
    );
}

/// Merges two inputs row by row.
#[divan::bench]
fn interleave(bencher: Bencher) {
    let first = Values::random(0, LEN / 2);
    let second = Values::random(1, LEN / 2);
    let array_indices: Buffer<u8> = (0..LEN / 2).flat_map(|_| [0, 1]).collect();
    let row_indices: Buffer<u32> = (0u32..).take(LEN / 2).flat_map(|row| [row, row]).collect();
    bench_case(
        bencher,
        || {
            (
                vec![first.widen(), second.widen()],
                index_array(&array_indices),
                index_array(&row_indices),
            )
        },
        |(values, array_indices, row_indices), ctx| {
            let interleaved = InterleaveArray::try_new(values, array_indices, row_indices);
            execute_canonical(interleaved.map(IntoArray::into_array), ctx)
        },
    );
}
