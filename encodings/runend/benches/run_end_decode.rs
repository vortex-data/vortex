// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;
use vortex_runend::decompress_bool::runend_decode_bools;
use vortex_runend::decompress_bool::runend_decode_typed_bool;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

/// Distribution types for bool benchmarks
#[derive(Clone, Copy)]
enum BoolDistribution {
    /// Alternating true/false (50/50)
    Alternating,
    /// Mostly true (90% true runs)
    MostlyTrue,
    /// Mostly false (90% false runs)
    MostlyFalse,
    /// All true
    AllTrue,
    /// All false
    AllFalse,
}

impl fmt::Display for BoolDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolDistribution::Alternating => write!(f, "alternating"),
            BoolDistribution::MostlyTrue => write!(f, "mostly_true"),
            BoolDistribution::MostlyFalse => write!(f, "mostly_false"),
            BoolDistribution::AllTrue => write!(f, "all_true"),
            BoolDistribution::AllFalse => write!(f, "all_false"),
        }
    }
}

#[derive(Clone, Copy)]
struct BoolBenchArgs {
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
}

impl fmt::Display for BoolBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}",
            self.total_length, self.avg_run_length, self.distribution
        )
    }
}

/// Creates bool test data with configurable distribution
fn create_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index.is_multiple_of(2),
            BoolDistribution::MostlyTrue => !run_index.is_multiple_of(10), // 90% true
            BoolDistribution::MostlyFalse => run_index.is_multiple_of(10), // 10% true (90% false)
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

// Medium size: 10k elements with various run lengths and distributions
const BOOL_ARGS: &[BoolBenchArgs] = &[
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::Alternating,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::MostlyTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::MostlyFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::AllTrue,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::AllFalse,
    },
    BoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::AllFalse,
    },
];

#[divan::bench(args = BOOL_ARGS)]
fn decode_bool(bencher: Bencher, args: BoolBenchArgs) {
    let BoolBenchArgs {
        total_length,
        avg_run_length,
        distribution,
    } = args;
    let (ends, values) = create_bool_test_data(total_length, avg_run_length, distribution);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}

#[derive(Clone, Copy)]
struct PredicateRunsBenchArgs {
    total_length: usize,
    source_run_length: usize,
    equal_result_runs: usize,
}

impl PredicateRunsBenchArgs {
    fn source_run_count(self) -> usize {
        self.total_length.div_ceil(self.source_run_length)
    }

    fn coalesced_run_count(self) -> usize {
        self.source_run_count().div_ceil(self.equal_result_runs)
    }
}

impl fmt::Display for PredicateRunsBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_rows_{}_source_runs_{}_coalesced_runs",
            self.total_length,
            self.source_run_count(),
            self.coalesced_run_count()
        )
    }
}

const PREDICATE_RUNS_ARGS: &[PredicateRunsBenchArgs] = &[
    PredicateRunsBenchArgs {
        total_length: 1_000_000,
        source_run_length: 8,
        equal_result_runs: 1,
    },
    PredicateRunsBenchArgs {
        total_length: 1_000_000,
        source_run_length: 8,
        equal_result_runs: 2,
    },
    PredicateRunsBenchArgs {
        total_length: 1_000_000,
        source_run_length: 8,
        equal_result_runs: 8,
    },
    PredicateRunsBenchArgs {
        total_length: 1_000_000,
        source_run_length: 8,
        equal_result_runs: 32,
    },
];

/// Models a predicate over numeric source runs. Multiple distinct source runs can produce the same
/// boolean result, leaving adjacent equal values in the RunEnd representation.
fn create_predicate_runs(args: PredicateRunsBenchArgs) -> (PrimitiveArray, BoolArray) {
    let source_run_count = args.source_run_count();
    let mut ends = BufferMut::<u32>::with_capacity(source_run_count);
    let mut values = Vec::with_capacity(source_run_count);

    for run_index in 0..source_run_count {
        ends.push(((run_index + 1) * args.source_run_length).min(args.total_length) as u32);
        values.push((run_index / args.equal_result_runs).is_multiple_of(2));
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

fn coalesce_adjacent_bool_runs_scalar(
    ends: &PrimitiveArray,
    values: &BoolArray,
) -> (PrimitiveArray, BoolArray) {
    let source_ends = ends.as_slice::<u32>();
    let source_values = values.to_bit_buffer();
    let mut coalesced_ends = BufferMut::<u32>::with_capacity(source_ends.len());
    let mut coalesced_values = Vec::with_capacity(source_ends.len());
    let mut previous_value = None;
    let mut previous_end = 0;

    for (&end, value) in source_ends.iter().zip(source_values.iter()) {
        if previous_value.is_some_and(|previous| previous != value) {
            coalesced_ends.push(previous_end);
        }
        if previous_value != Some(value) {
            coalesced_values.push(value);
        }
        previous_value = Some(value);
        previous_end = end;
    }

    if let Some(&last_end) = source_ends.last() {
        coalesced_ends.push(last_end);
    }

    (
        PrimitiveArray::new(coalesced_ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(coalesced_values)),
    )
}

fn coalesce_adjacent_bool_runs_wordwise(
    ends: &PrimitiveArray,
    values: &BoolArray,
) -> (PrimitiveArray, BoolArray) {
    let source_ends = ends.as_slice::<u32>();
    let source_values = values.to_bit_buffer();
    if source_values.is_empty() {
        return (
            PrimitiveArray::new(BufferMut::<u32>::empty().freeze(), Validity::NonNullable),
            BoolArray::from(BitBuffer::empty()),
        );
    }

    let mut coalesced_ends = BufferMut::<u32>::with_capacity(source_ends.len());
    let first_value = source_values.value(0);
    let mut previous_value = first_value;
    let word_count = source_values.len().div_ceil(64);

    for (word_index, word) in source_values
        .chunks()
        .iter_padded()
        .take(word_count)
        .enumerate()
    {
        let word_start = word_index * 64;
        let valid_bits = (source_values.len() - word_start).min(64);
        let preceding_bits = (word << 1) | u64::from(previous_value);
        let mut transitions = word ^ preceding_bits;
        if valid_bits < 64 {
            transitions &= (1_u64 << valid_bits) - 1;
        }
        if word_index == 0 {
            transitions &= !1;
        }

        while transitions != 0 {
            let bit_index = transitions.trailing_zeros() as usize;
            coalesced_ends.push(source_ends[word_start + bit_index - 1]);
            transitions &= transitions - 1;
        }

        previous_value = ((word >> (valid_bits - 1)) & 1) != 0;
    }

    coalesced_ends.push(source_ends[source_ends.len() - 1]);
    let coalesced_values = BitBuffer::collect_bool(coalesced_ends.len(), |index| {
        first_value ^ !index.is_multiple_of(2)
    });

    (
        PrimitiveArray::new(coalesced_ends.freeze(), Validity::NonNullable),
        BoolArray::from(coalesced_values),
    )
}

fn count_bool_runs_wordwise(values: &BitBuffer) -> usize {
    if values.is_empty() {
        return 0;
    }

    let mut transitions = 0usize;
    let mut previous_value = values.value(0);
    let word_count = values.len().div_ceil(64);
    for (word_index, word) in values.chunks().iter_padded().take(word_count).enumerate() {
        let word_start = word_index * 64;
        let valid_bits = (values.len() - word_start).min(64);
        let preceding_bits = (word << 1) | u64::from(previous_value);
        let mut transition_word = word ^ preceding_bits;
        if valid_bits < 64 {
            transition_word &= (1_u64 << valid_bits) - 1;
        }
        if word_index == 0 {
            transition_word &= !1;
        }
        transitions += transition_word.count_ones() as usize;
        previous_value = ((word >> (valid_bits - 1)) & 1) != 0;
    }
    transitions + 1
}

fn assert_wordwise_coalescing_matches_scalar(ends: &PrimitiveArray, values: &BoolArray) {
    let (scalar_ends, scalar_values) = coalesce_adjacent_bool_runs_scalar(ends, values);
    let (wordwise_ends, wordwise_values) = coalesce_adjacent_bool_runs_wordwise(ends, values);
    assert_eq!(
        scalar_ends.as_slice::<u32>(),
        wordwise_ends.as_slice::<u32>()
    );
    assert_eq!(
        scalar_values.to_bit_buffer(),
        wordwise_values.to_bit_buffer()
    );
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn decode_predicate_runs_raw(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            let values = values.to_bit_buffer();
            runend_decode_typed_bool(
                ends.as_slice::<u32>().iter().map(|&end| end as usize),
                &values,
                Mask::AllTrue(values.len()),
                Nullability::NonNullable,
                args.total_length,
            )
        });
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn decode_predicate_runs_adaptive(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, args.total_length, ctx)
        });
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn coalesce_then_decode_predicate_runs(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            let (coalesced_ends, coalesced_values) =
                coalesce_adjacent_bool_runs_scalar(ends, values);
            runend_decode_bools(coalesced_ends, coalesced_values, 0, args.total_length, ctx)
        });
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn coalesce_then_decode_predicate_runs_wordwise(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    assert_wordwise_coalescing_matches_scalar(&ends, &values);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            let (coalesced_ends, coalesced_values) =
                coalesce_adjacent_bool_runs_wordwise(ends, values);
            runend_decode_bools(coalesced_ends, coalesced_values, 0, args.total_length, ctx)
        });
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn adaptive_coalesce_then_decode_predicate_runs(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    assert_wordwise_coalescing_matches_scalar(&ends, &values);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            let values_buffer = values.to_bit_buffer();
            if count_bool_runs_wordwise(&values_buffer) * 2 <= values_buffer.len() {
                let (coalesced_ends, coalesced_values) =
                    coalesce_adjacent_bool_runs_wordwise(ends, values);
                runend_decode_bools(coalesced_ends, coalesced_values, 0, args.total_length, ctx)
            } else {
                runend_decode_bools(ends.clone(), values.clone(), 0, args.total_length, ctx)
            }
        });
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn coalesce_predicate_runs_scalar(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| coalesce_adjacent_bool_runs_scalar(ends, values));
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn coalesce_predicate_runs_wordwise(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (ends, values) = create_predicate_runs(args);
    assert_wordwise_coalescing_matches_scalar(&ends, &values);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| coalesce_adjacent_bool_runs_wordwise(ends, values));
}

#[divan::bench(args = PREDICATE_RUNS_ARGS)]
fn count_predicate_runs_wordwise(bencher: Bencher, args: PredicateRunsBenchArgs) {
    let (_, values) = create_predicate_runs(args);
    let values = values.to_bit_buffer();
    assert_eq!(
        count_bool_runs_wordwise(&values),
        args.coalesced_run_count()
    );
    bencher.bench_local(|| count_bool_runs_wordwise(&values));
}

/// Validity distribution for nullable benchmarks
#[derive(Clone, Copy)]
enum ValidityDistribution {
    /// 90% valid
    MostlyValid,
    /// 50% valid
    HalfValid,
    /// 10% valid
    MostlyNull,
}

impl fmt::Display for ValidityDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidityDistribution::MostlyValid => write!(f, "mostly_valid"),
            ValidityDistribution::HalfValid => write!(f, "half_valid"),
            ValidityDistribution::MostlyNull => write!(f, "mostly_null"),
        }
    }
}

#[derive(Clone, Copy)]
struct NullableBoolBenchArgs {
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity: ValidityDistribution,
}

impl fmt::Display for NullableBoolBenchArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{}_{}_{}",
            self.total_length, self.avg_run_length, self.distribution, self.validity
        )
    }
}

/// Creates nullable bool test data with configurable distribution and validity
fn create_nullable_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
    validity: ValidityDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);
    let mut validity_bits = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index.is_multiple_of(2),
            BoolDistribution::MostlyTrue => !run_index.is_multiple_of(10),
            BoolDistribution::MostlyFalse => run_index.is_multiple_of(10),
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);

        let is_valid = match validity {
            ValidityDistribution::MostlyValid => !run_index.is_multiple_of(10),
            ValidityDistribution::HalfValid => run_index.is_multiple_of(2),
            ValidityDistribution::MostlyNull => run_index.is_multiple_of(10),
        };
        validity_bits.push(is_valid);

        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::new(
            BitBuffer::from(values),
            Validity::from(BitBuffer::from(validity_bits)),
        ),
    )
}

const NULLABLE_BOOL_ARGS: &[NullableBoolBenchArgs] = &[
    // Alternating with different validity
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::HalfValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyNull,
    },
    // MostlyTrue with different validity
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::HalfValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 10,
        distribution: BoolDistribution::MostlyTrue,
        validity: ValidityDistribution::MostlyNull,
    },
    // Different run lengths with MostlyValid
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 2,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 100,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
    NullableBoolBenchArgs {
        total_length: 10_000,
        avg_run_length: 1000,
        distribution: BoolDistribution::Alternating,
        validity: ValidityDistribution::MostlyValid,
    },
];

#[divan::bench(args = NULLABLE_BOOL_ARGS)]
fn decode_bool_nullable(bencher: Bencher, args: NullableBoolBenchArgs) {
    let NullableBoolBenchArgs {
        total_length,
        avg_run_length,
        distribution,
        validity,
    } = args;
    let (ends, values) =
        create_nullable_bool_test_data(total_length, avg_run_length, distribution, validity);
    bencher
        .with_inputs(|| (ends.clone(), values.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(ends, values, ctx)| {
            runend_decode_bools(ends.clone(), values.clone(), 0, total_length, ctx)
        });
}
