// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use num_traits::AsPrimitive;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use super::*;

/// Every statistic, computed naively over the valid values.
#[derive(Debug, PartialEq)]
struct Naive {
    min_max: (i32, i32),
    runs: u32,
    counts: HashMap<i32, u32>,
    sorted: SortedResult,
    sum: i128,
    and_or: (u64, u64),
    bit_widths: Vec<u32>,
    deltas: Option<(i128, i128)>,
}

fn naive(values: &[i32], valid: &[bool]) -> Option<Naive> {
    let v: Vec<i32> = values
        .iter()
        .zip(valid)
        .filter(|(_, ok)| **ok)
        .map(|(v, _)| *v)
        .collect();
    let mut counts = HashMap::default();
    let mut bit_widths = vec![0; 33];
    for &x in &v {
        *counts.entry(x).or_insert(0) += 1;
        bit_widths[32 - x.leading_zeros() as usize] += 1;
    }
    let deltas: Vec<i128> = v
        .windows(2)
        .map(|w| i128::from(w[1]) - i128::from(w[0]))
        .collect();
    Some(Naive {
        min_max: (*v.iter().min()?, *v.iter().max()?),
        runs: 1 + v.windows(2).filter(|w| w[0] != w[1]).count() as u32,
        counts,
        sorted: SortedResult {
            sorted: v.is_sorted(),
            strict_sorted: v.windows(2).all(|w| w[0] < w[1]),
        },
        sum: v.iter().map(|&x| i128::from(x)).sum(),
        and_or: (
            u64::from(v.iter().fold(u32::MAX, |acc, &x| acc & x as u32)),
            u64::from(v.iter().fold(0, |acc, &x| acc | x as u32)),
        ),
        bit_widths,
        deltas: deltas
            .iter()
            .min()
            .zip(deltas.iter().max())
            .map(|(a, b)| (*a, *b)),
    })
}

/// Computes every statistic in one fused pass. Tuples nest, so the composition can exceed the
/// largest tuple arity.
fn fused(values: &[i32], validity: &Mask, min: i32, max: i32) -> Option<Naive> {
    let (((min, max), runs, (distinct, distinct_runs), sorted), (sum, bits, widths, deltas)) =
        accumulate(
            values,
            validity,
            (
                (
                    MinMax::new(),
                    RunCount::new(),
                    Distinct::new(min, max, values.len()),
                    Sorted::new(),
                ),
                (
                    Sum::new(),
                    CommonBits::new(),
                    BitWidthHistogram::new(),
                    DeltaRange::new(),
                ),
            ),
        )?;
    assert_eq!(distinct_runs, runs);
    Some(Naive {
        min_max: (min, max),
        runs,
        counts: distinct
            .distinct_values()
            .iter()
            .map(|(k, &c)| (k.0, c))
            .collect(),
        sorted,
        sum,
        and_or: (bits.and, bits.or),
        bit_widths: widths,
        deltas,
    })
}

#[rstest]
fn fused_matches_naive(
    #[values(0, 1, 63, 64, 65, 64 * 20 + 17)] len: usize,
    #[values(1, 3, 64, 100)] run_len: usize,
    #[values(None, Some(1), Some(3), Some(97), Some(usize::MAX))] null_every: Option<usize>,
    #[values(16, 100_000)] range: i32,
    #[values(false, true)] invert_validity: bool,
) {
    let values: Vec<i32> = (0..len)
        .map(|i| ((i / run_len) as i32).wrapping_mul(7919) % range - range / 2)
        .collect();
    // `Some(usize::MAX)` makes only the first value null; `Some(1)` makes every value null.
    let valid: Vec<bool> = (0..len)
        .map(|i| null_every.is_none_or(|n| (i % n != 0) != invert_validity))
        .collect();
    let validity = match null_every {
        None => Mask::new_true(len),
        Some(_) => Mask::from_iter(valid.iter().copied()),
    };

    let expected = naive(&values, &valid);
    let actual = expected
        .as_ref()
        .and_then(|e| fused(&values, &validity, e.min_max.0, e.min_max.1));
    assert_eq!(actual, expected);
}

#[rstest]
fn sorted_inputs(
    #[values(1, 2, 64, 65, 200)] len: i32,
    #[values(None, Some(5))] null_every: Option<i32>,
) {
    let validity = Mask::from_iter((0..len).map(|i| null_every.is_none_or(|n| i % n != 1)));
    let valid: Vec<bool> = validity.to_bit_buffer().iter().collect();
    let strict: Vec<i32> = (0..len).collect();
    let repeated: Vec<i32> = (0..len).map(|i| i / 2).collect();
    let mut descending = strict.clone();
    descending.reverse();
    for values in [strict, repeated, descending] {
        assert_eq!(
            accumulate(&values, &validity, Sorted::new()),
            naive(&values, &valid).map(|n| n.sorted),
            "{values:?}"
        );
    }
}

#[test]
fn full_domain_distinct_needs_no_bounds() {
    let values: Vec<i8> = (i8::MIN..=i8::MAX).chain([i8::MAX, 0]).collect();
    let (distinct, runs) = accumulate(
        &values,
        &Mask::new_true(values.len()),
        Distinct::for_full_domain(values.len()).unwrap(),
    )
    .unwrap();
    assert_eq!(distinct.distinct_count(), 256);
    assert_eq!(distinct.most_frequent(), (i8::MAX, 2));
    assert_eq!(runs, 257);
    assert!(Distinct::<i32>::for_full_domain(10).is_none());
}

#[rstest]
#[case::u8_max(vec![u8::MAX; 50_000], 255 * 50_000)]
#[case::i8_min(vec![i8::MIN; 50_000], -128 * 50_000)]
#[case::i8_max(vec![i8::MAX; 50_000], 127 * 50_000)]
#[case::u16_max(vec![u16::MAX; 50_000], 65_535 * 50_000)]
#[case::i16_min(vec![i16::MIN; 50_000], -32_768 * 50_000)]
#[case::u32_max(vec![u32::MAX; 50_000], (u32::MAX as i128) * 50_000)]
#[case::u64_max(vec![u64::MAX; 1000], (u64::MAX as i128) * 1000)]
#[case::i64_min(vec![i64::MIN; 1000], (i64::MIN as i128) * 1000)]
#[case::i64_mixed((0..1000).map(|i| if i % 2 == 0 { i64::MAX } else { -3 }).collect(), (i64::MAX as i128) * 500 - 1500)]
fn sum_is_exact<T: IntValue>(#[case] values: Vec<T>, #[case] expected: i128) {
    let validity = Mask::new_true(values.len());
    assert_eq!(accumulate(&values, &validity, Sum::new()), Some(expected));
}

#[test]
fn bits_use_the_bit_pattern() {
    let values = [-1i8, 4];
    let bits = accumulate(&values, &Mask::new_true(2), CommonBits::new()).unwrap();
    assert_eq!((bits.and, bits.or), (0x04, 0xFF));
    assert_eq!(bits.max_bit_width(), 8);

    let values = [8u16, 24, 40];
    let bits = accumulate(&values, &Mask::new_true(3), CommonBits::new()).unwrap();
    assert_eq!(bits.trailing_zeros(), 3);
    assert_eq!(bits.max_bit_width(), 6);
    assert_eq!(bits.constant_bits(), !0b11_0000 & 0xFFFF);
}

#[test]
fn wide_deltas_are_exact() {
    let values = [i64::MIN, i64::MAX, i64::MIN];
    let deltas = accumulate(&values, &Mask::new_true(3), DeltaRange::new()).unwrap();
    assert_eq!(deltas, Some((-(u64::MAX as i128), u64::MAX as i128)));
    assert_eq!(
        accumulate(&[7u8], &Mask::new_true(1), DeltaRange::new()),
        Some(None)
    );
}

/// The erased set is filled by a pass selected per integer type, and read without knowing it.
#[test]
fn erased_stats() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = PrimitiveArray::new(
        buffer![10u16, 20, 30, 40, 999],
        Validity::from_iter([true, true, true, true, false]),
    );
    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), &mut ctx)?;

    let stats = match_each_integer_ptype!(array.ptype(), |T| {
        compute(
            array.as_slice::<T>(),
            &validity,
            (MinMax::new(), Sum::new(), DeltaRange::new(), Sorted::new()),
        )
    });

    assert_eq!(stats.len(), 4);
    assert_eq!(
        stats.get::<MinMaxStat>(),
        Some(&(PValue::U16(10), PValue::U16(40)))
    );
    assert_eq!(stats.get::<SumStat>(), Some(&100));
    assert_eq!(stats.get::<DeltaRangeStat>(), Some(&Some((10, 10))));
    assert_eq!(
        stats.get::<SortedStat>().map(|s| s.strict_sorted),
        Some(true)
    );
    assert!(stats.get::<DistinctStat>().is_none());

    let none = compute(&[1u8, 2], &Mask::new_false(2), MinMax::new());
    assert!(none.is_empty());
    Ok(())
}

/// Every statistic of the valid values computed naively, in a type-independent form.
fn naive_generic<T: IntValue>(values: &[T], valid: &[bool]) -> Option<GenericStats> {
    let v: Vec<T> = values
        .iter()
        .zip(valid)
        .filter(|(_, ok)| **ok)
        .map(|(v, _)| *v)
        .collect();
    let width = 8 * size_of::<T>();
    let mut widths = vec![0u32; width + 1];
    for &x in &v {
        let bits: u64 = x.as_();
        let bits = if width == 64 {
            bits
        } else {
            bits & ((1 << width) - 1)
        };
        widths[(64 - bits.leading_zeros()) as usize] += 1;
    }
    let deltas: Vec<i128> = v
        .windows(2)
        .map(|w| AsPrimitive::<i128>::as_(w[1]) - AsPrimitive::<i128>::as_(w[0]))
        .collect();
    Some(GenericStats {
        min_max: (v.iter().min()?.to_pvalue(), v.iter().max()?.to_pvalue()),
        runs: 1 + v.windows(2).filter(|w| w[0] != w[1]).count() as u32,
        sorted: v.is_sorted(),
        sum: v.iter().map(|&x| AsPrimitive::<i128>::as_(x)).sum(),
        widths,
        deltas: deltas
            .iter()
            .min()
            .zip(deltas.iter().max())
            .map(|(a, b)| (*a, *b)),
    })
}

/// The statistics compared by [`compositions_match_naive`].
#[derive(Debug, PartialEq)]
struct GenericStats {
    min_max: (PValue, PValue),
    runs: u32,
    sorted: bool,
    sum: i128,
    widths: Vec<u32>,
    deltas: Option<(i128, i128)>,
}

fn generic_stats(stats: &IntStats) -> Option<GenericStats> {
    Some(GenericStats {
        min_max: *stats.get::<MinMaxStat>()?,
        runs: *stats.get::<RunCountStat>()?,
        sorted: stats.get::<SortedStat>()?.sorted,
        sum: *stats.get::<SumStat>()?,
        widths: stats.get::<BitWidthHistogramStat>()?.clone(),
        deltas: *stats.get::<DeltaRangeStat>()?,
    })
}

/// Checks every schedule of the same statistics against a naive reference across block
/// boundaries: one loop each, one loop for all, groups in between, and a plain nested tuple.
fn check_compositions<T: IntValue>(values: Vec<T>, valid: Vec<bool>) {
    let validity = if valid.iter().all(|&v| v) {
        Mask::new_true(values.len())
    } else {
        Mask::from_iter(valid.iter().copied())
    };
    let expected = naive_generic(&values, &valid);
    let stats = || {
        (
            MinMax::new(),
            Sum::new(),
            CommonBits::new(),
            RunCount::new(),
            Sorted::new(),
            BitWidthHistogram::new(),
            DeltaRange::new(),
        )
    };

    macro_rules! check {
        ($($label:literal => $schedule:expr),+ $(,)?) => {
            $(assert_eq!(
                generic_stats(&compute(&values, &validity, Schedule::<_, { $schedule }>::new(stats()))),
                expected,
                $label,
            );)+
        };
    }
    check!(
        "each" => EACH,
        "fused" => FUSED,
        "cheap fused" => groups(&[3, 4, 5, 6]),
        "two groups" => groups(&[3]),
        "planned" => groups(&[3, 5, 6]),
    );

    let (cheap, runs, sorted, widths, deltas) = (
        Schedule::<_, FUSED>::new((MinMax::new(), Sum::new(), CommonBits::new())),
        RunCount::new(),
        Sorted::new(),
        BitWidthHistogram::new(),
        DeltaRange::new(),
    );
    let nested = compute(&values, &validity, (cheap, runs, sorted, widths, deltas));
    assert_eq!(generic_stats(&nested), expected, "nested");
}

/// Values with runs, spanning several blocks of every type, plus a tail.
fn wide_values(len: usize) -> Vec<i64> {
    (0..len as i64)
        .map(|i| (i / 3).wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as i64) ^ (i >> 7))
        .collect()
}

#[rstest]
fn compositions_match_naive(
    #[values(0, 1, 63, 64 * 300 + 5, 40_000)] len: usize,
    #[values(None, Some(10), Some(2))] null_every: Option<usize>,
    #[values(false, true)] null_blocks: bool,
) {
    // Optionally null out whole blocks, so that some blocks are skipped entirely.
    let valid: Vec<bool> = (0..len)
        .map(|i| null_every.is_none_or(|n| i % n != 0) && !(null_blocks && (i / 9000) % 2 == 1))
        .collect();
    let raw = wide_values(len);
    macro_rules! check {
        ($($T:ty),+) => {
            $(check_compositions::<$T>(
                raw.iter().map(|&v| v as $T).collect(),
                valid.clone(),
            );)+
        };
    }
    check!(u8, i8, u16, i16, u32, i32, u64, i64);
    // Narrow values exercise the low histogram buckets and dense equal runs.
    check!(u8, u16);
    check_compositions::<u8>(
        raw.iter().map(|&v| (v & 0x7) as u8).collect(),
        valid.clone(),
    );
    check_compositions::<u16>(raw.iter().map(|&v| (v & 0x3FF) as u16).collect(), valid);
}
