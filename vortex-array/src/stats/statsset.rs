use enum_iterator::{all, Sequence};
use itertools::{EitherOrBoth, Itertools};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexError, VortexExpect, VortexUnwrap};
use vortex_scalar::{Scalar, ScalarValue};

use crate::stats::{
    exact, GtOrd, Max, Min, NullCount, PartialOrder, Precision, Stat, StatOrder, StatisticsCompare,
    TrueCount, UncompressedSizeInBytes,
};

#[derive(Default, Debug, Clone)]
pub struct StatsSet {
    values: Option<Vec<(Stat, Precision<ScalarValue>)>>,
}

impl StatsSet {
    /// Create new StatSet without validating uniqueness of all the entries
    ///
    /// # Safety
    ///
    /// This method will not panic or trigger UB, but may lead to duplicate stats being stored.
    pub fn new_unchecked(values: Vec<(Stat, Precision<ScalarValue>)>) -> Self {
        Self {
            values: Some(values),
        }
    }

    /// Specialized constructor for the case where the StatsSet represents
    /// an array consisting entirely of [null](vortex_dtype::DType::Null) values.
    pub fn nulls(len: usize, dtype: &DType) -> Self {
        let mut stats = Self::new_unchecked(vec![
            (Stat::RunCount, exact(1)),
            (Stat::NullCount, exact(len)),
        ]);

        if len > 0 {
            stats.set(Stat::IsConstant, exact(true));
            stats.set(Stat::IsSorted, exact(true));
            stats.set(Stat::IsStrictSorted, exact(len < 2));
        }

        // Add any DType-specific stats.
        match dtype {
            DType::Bool(_) => {
                stats.set(Stat::TrueCount, exact(0));
            }
            DType::Primitive(ptype, _) => {
                ptype.byte_width();
                stats.set(
                    Stat::BitWidthFreq,
                    exact(vec![0u64; ptype.byte_width() * 8 + 1]),
                );
                stats.set(
                    Stat::TrailingZeroFreq,
                    exact(vec![
                        ptype.byte_width() as u64 * 8;
                        ptype.byte_width() * 8 + 1
                    ]),
                );
            }
            _ => {}
        }

        stats
    }

    pub fn constant(scalar: Scalar, length: usize) -> Self {
        let (dtype, sv) = scalar.into_parts();
        let mut stats = Self::default();
        if length > 0 {
            stats.extend([
                (Stat::IsConstant, exact(true)),
                (Stat::IsSorted, exact(true)),
                (Stat::IsStrictSorted, exact(length <= 1)),
            ]);
        }

        let run_count = if length == 0 { 0u64 } else { 1 };
        stats.set(Stat::RunCount, exact(run_count));

        let null_count = if sv.is_null() { length as u64 } else { 0 };
        stats.set(Stat::NullCount, exact(null_count));

        if !sv.is_null() {
            stats.extend([
                (Stat::Min, exact(sv.clone())),
                (Stat::Max, exact(sv.clone())),
            ]);
        }

        if matches!(dtype, DType::Bool(_)) {
            let bool_val = <Option<bool>>::try_from(&sv).vortex_expect("Checked dtype");
            let true_count = bool_val
                .map(|b| if b { length as u64 } else { 0 })
                .unwrap_or(0);
            stats.set(Stat::TrueCount, exact(true_count));
        }

        stats
    }

    pub fn bools_with_true_and_null_count(
        true_count: usize,
        null_count: usize,
        len: usize,
    ) -> Self {
        StatsSet::new_unchecked(vec![
            (Stat::TrueCount, exact(true_count)),
            (Stat::NullCount, exact(null_count)),
            (Stat::Min, exact(true_count == len)),
            (Stat::Max, exact(true_count > 0)),
            (
                Stat::IsConstant,
                exact((true_count == 0 && null_count == 0) || true_count == len),
            ),
        ])
    }

    pub fn of(stat: Stat, value: Precision<ScalarValue>) -> Self {
        Self::new_unchecked(vec![(stat, value)])
    }
}

// Getters and setters for individual stats.
impl StatsSet {
    /// Count of stored stats with known values.
    pub fn len(&self) -> usize {
        self.values.as_ref().map_or(0, |v| v.len())
    }

    /// Predicate equivalent to a [len][Self::len] of zero.
    pub fn is_empty(&self) -> bool {
        self.values.as_ref().is_none_or(|v| v.is_empty())
    }

    pub fn get(&self, stat: Stat) -> Option<&Precision<ScalarValue>> {
        self.values
            .as_ref()
            .and_then(|v| v.iter().find(|(s, _)| *s == stat).map(|(_, v)| v))
    }

    pub fn getb<S>(&self, dtype: &DType) -> Option<S::BoundDirection>
    where
        // U: for<'b> TryFrom<&'b ScalarValue, Error = VortexError> + PartialOrd,
        S: StatOrder<Scalar>,
    {
        self.get(S::STAT)
            .map(|s| S::BoundDirection::lift(s.as_ref().into_scalar(dtype.clone())))
    }

    pub fn getv<S>(&self, dtype: &DType) -> Option<Precision<Scalar>>
    where
        S: StatOrder<Scalar>,
    {
        self.get(S::STAT)
            .map(|s| s.as_ref().into_scalar(dtype.clone()))
    }

    pub fn get_asv<S, T>(&self) -> Option<Precision<T>>
    where
        T: for<'a> TryFrom<&'a ScalarValue, Error = VortexError> + PartialOrd,
        S: StatOrder<T>,
    {
        self.get_as::<T>(S::STAT)
    }

    pub fn get_as<T: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<Precision<T>> {
        self.get(stat).map(|v| {
            v.as_ref().map(|v| {
                T::try_from(v).unwrap_or_else(|err| {
                    vortex_panic!(
                        err,
                        "Failed to get stat {} as {}",
                        stat,
                        std::any::type_name::<T>()
                    )
                })
            })
        })
    }

    /// Set the stat `stat` to `value`.
    pub fn set(&mut self, stat: Stat, value: Precision<ScalarValue>) {
        if self.values.is_none() {
            self.values = Some(Vec::with_capacity(Stat::CARDINALITY));
        }
        let values = self.values.as_mut().vortex_expect("we just initialized it");
        if let Some(existing) = values.iter_mut().find(|(s, _)| *s == stat) {
            *existing = (stat, value);
        } else {
            values.push((stat, value));
        }
    }

    /// Clear the stat `stat` from the set.
    pub fn clear(&mut self, stat: Stat) {
        if let Some(v) = &mut self.values {
            v.retain(|(s, _)| *s != stat);
        }
    }

    pub fn retain_only(&mut self, stats: &[Stat]) {
        if let Some(v) = &mut self.values {
            v.retain(|(s, _)| stats.contains(s));
        }
    }

    /// Iterate over the statistic names and values in-place.
    ///
    /// See [Iterator].
    pub fn iter(&self) -> impl Iterator<Item = &(Stat, Precision<ScalarValue>)> {
        self.values.iter().flat_map(|v| v.iter())
    }
}

// StatSetIntoIter just exists to protect current implementation from exposure on the public API.

/// Owned iterator over the stats.
///
/// See [IntoIterator].
pub struct StatsSetIntoIter(Option<std::vec::IntoIter<(Stat, Precision<ScalarValue>)>>);

impl Iterator for StatsSetIntoIter {
    type Item = (Stat, Precision<ScalarValue>);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(|i| i.next())
    }
}

impl IntoIterator for StatsSet {
    type Item = (Stat, Precision<ScalarValue>);
    type IntoIter = StatsSetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        StatsSetIntoIter(self.values.map(|v| v.into_iter()))
    }
}

impl FromIterator<(Stat, Precision<ScalarValue>)> for StatsSet {
    fn from_iter<T: IntoIterator<Item = (Stat, Precision<ScalarValue>)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut this = Self {
            values: Some(Vec::with_capacity(lower_bound)),
        };
        this.extend(iter);
        this
    }
}

impl Extend<(Stat, Precision<ScalarValue>)> for StatsSet {
    #[inline]
    fn extend<T: IntoIterator<Item = (Stat, Precision<ScalarValue>)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        if let Some(v) = &mut self.values {
            v.reserve(lower_bound);
        }
        iter.for_each(|(stat, value)| self.set(stat, value));
    }
}

// Merge helpers
impl StatsSet {
    /// Merge stats set `other` into `self`, with the semantic assumption that `other`
    /// contains stats from an array that is *appended* to the array represented by `self`.
    pub fn merge_ordered(mut self, other: &Self, dtype: &DType) -> Self {
        for s in all::<Stat>() {
            match s {
                Stat::BitWidthFreq => self.merge_bit_width_freq(other),
                Stat::TrailingZeroFreq => self.merge_trailing_zero_freq(other),
                Stat::IsConstant => self.merge_is_constant(other, dtype),
                Stat::IsSorted => self.merge_is_sorted(other, dtype),
                Stat::IsStrictSorted => self.merge_is_strict_sorted(other, dtype),
                Stat::Max => self.merge_max(other, dtype),
                Stat::Min => self.merge_min(other, dtype),
                Stat::RunCount => self.merge_run_count(other),
                Stat::TrueCount => self.merge_true_count(other),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
            }
        }

        self
    }

    /// Merge stats set `other` into `self`, with no assumption on ordering.
    /// Stats that are not commutative (e.g., is_sorted) are dropped from the result.
    pub fn merge_unordered(mut self, other: &Self, dtype: &DType) -> Self {
        for s in all::<Stat>() {
            if !s.is_commutative() {
                self.clear(s);
                continue;
            }

            match s {
                Stat::BitWidthFreq => self.merge_bit_width_freq(other),
                Stat::TrailingZeroFreq => self.merge_trailing_zero_freq(other),
                Stat::IsConstant => self.merge_is_constant(other, dtype),
                Stat::Max => self.merge_max(other, dtype),
                Stat::Min => self.merge_min(other, dtype),
                Stat::TrueCount => self.merge_true_count(other),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
                _ => vortex_panic!("Unrecognized commutative stat {}", s),
            }
        }

        self
    }

    fn merge_min(&mut self, other: &Self, dtype: &DType) {
        match (self.getb::<Min>(dtype), other.getb::<Min>(dtype)) {
            (Some(m1), Some(m2)) => {
                if m2.le(&m1).vortex_expect("can compare min stats") {
                    self.set(Stat::Min, m2.into_value().map(|s| s.into_value()));
                }
            }
            _ => self.clear(Stat::Min),
        }
    }

    fn merge_max(&mut self, other: &Self, dtype: &DType) {
        match (self.getb::<Max>(dtype), other.getb::<Max>(dtype)) {
            (Some(m1), Some(m2)) => {
                if m2.ge(&m1).vortex_expect("can compare max stats") {
                    self.set(Stat::Max, m2.into_value().map(|s| s.into_value()));
                }
            }
            _ => self.clear(Stat::Max),
        }
    }

    fn merge_is_constant(&mut self, other: &Self, dtype: &DType) {
        if (Some(Precision::Exact(true)), Some(Precision::Exact(true)))
            == (
                self.get_as(Stat::IsConstant),
                other.get_as(Stat::IsConstant),
            )
            && self.getv::<Min>(dtype) == other.getv::<Min>(dtype)
        {
            return;
        }
        // TODO(joe): this is not true, what is the correct thing to do here? Maybe bound(false)?
        self.set(Stat::IsConstant, exact(false));
    }

    fn merge_is_sorted(&mut self, other: &Self, dtype: &DType) {
        self.merge_sortedness_stat(other, Stat::IsSorted, dtype, PartialOrd::le)
    }

    fn merge_is_strict_sorted(&mut self, other: &Self, dtype: &DType) {
        self.merge_sortedness_stat(other, Stat::IsStrictSorted, dtype, PartialOrd::lt)
    }

    fn merge_sortedness_stat<F: Fn(&Scalar, &Scalar) -> bool>(
        &mut self,
        other: &Self,
        stat: Stat,
        dtype: &DType,
        cmp: F,
    ) {
        if (Some(Precision::Exact(true)), Some(Precision::Exact(true)))
            == (self.get_as(stat), other.get_as(stat))
        {
            // There might be no stat because it was dropped, or it doesn't exist
            // (e.g. an all null array).
            // We assume that it was the dropped case since the doesn't exist might imply sorted,
            // but this in-precision is correct.
            if let (Some(self_max), Some(other_min)) =
                (self.getv::<Max>(dtype), other.getv::<Min>(dtype))
            {
                return if cmp(self_max.value(), other_min.value()) {
                    // keep value
                } else {
                    // TODO(joe): this might not be false, I guess this might be a bound.
                    self.set(stat, exact(false));
                };
            }
        }
        self.clear(stat);
    }

    fn merge_true_count(&mut self, other: &Self) {
        self.merge_sum_stat::<TrueCount>(other)
    }

    fn merge_null_count(&mut self, other: &Self) {
        self.merge_sum_stat::<NullCount>(other)
    }

    fn merge_uncompressed_size_in_bytes(&mut self, other: &Self) {
        self.merge_sum_stat::<UncompressedSizeInBytes>(other)
    }

    fn merge_sum_stat<S: StatOrder<usize>>(&mut self, other: &Self) {
        match (self.get_asv::<S, usize>(), other.get_asv::<S, usize>()) {
            (Some(nc1), Some(nc2)) => {
                self.set(
                    S::STAT,
                    nc1.and_then_prefer_bound(|nc1| nc2.map(|nc2| ScalarValue::from(nc1 + nc2))),
                );
            }
            _ => self.clear(S::STAT),
        }
    }

    fn merge_bit_width_freq(&mut self, other: &Self) {
        self.merge_freq_stat(other, Stat::BitWidthFreq)
    }

    fn merge_trailing_zero_freq(&mut self, other: &Self) {
        self.merge_freq_stat(other, Stat::TrailingZeroFreq)
    }

    fn merge_freq_stat(&mut self, other: &Self, stat: Stat) {
        match (
            self.get_as::<Vec<usize>>(stat),
            other.get_as::<Vec<usize>>(stat),
        ) {
            (Some(f1), Some(f2)) => {
                let combined_freq = f1.and_then_prefer_bound(|f1| {
                    f2.map(|f2| {
                        ScalarValue::from(
                            f1.iter()
                                .zip_longest(f2.iter())
                                .map(|pair| match pair {
                                    EitherOrBoth::Both(a, b) => a + b,
                                    EitherOrBoth::Left(v) | EitherOrBoth::Right(v) => *v,
                                })
                                .collect_vec(),
                        )
                    })
                });
                self.set(stat, combined_freq);
            }
            _ => self.clear(stat),
        }
    }

    /// Merged run count is an upper bound where we assume run is interrupted at the boundary
    fn merge_run_count(&mut self, other: &Self) {
        match (
            self.get_as::<usize>(Stat::RunCount),
            other.get_as::<usize>(Stat::RunCount),
        ) {
            (Some(r1), Some(r2)) => {
                self.set(
                    Stat::RunCount,
                    r1.and_then_prefer_bound(|r1| r2.map(|r2| ScalarValue::from(r1 + r2 + 1))),
                );
            }
            _ => self.clear(Stat::RunCount),
        }
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;
    use itertools::Itertools;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::array::PrimitiveArray;
    use crate::stats::{bound, exact, ArrayStatistics as _, Stat, StatsSet};
    use crate::IntoArrayData as _;

    #[test]
    fn test_iter() {
        let set = StatsSet::new_unchecked(vec![(Stat::Max, exact(100)), (Stat::Min, exact(42))]);
        let mut iter = set.iter();
        let first = iter.next().unwrap().clone();
        assert_eq!(first.0, Stat::Max);
        assert_eq!(first.1.map(|f| i32::try_from(&f).unwrap()), exact(100));
        let snd = iter.next().unwrap().clone();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(snd.1.map(|s| i32::try_from(&s).unwrap()), 42);
    }

    #[test]
    fn into_iter() {
        let mut set =
            StatsSet::new_unchecked(vec![(Stat::Max, exact(100)), (Stat::Min, exact(42))])
                .into_iter();
        let (stat, first) = set.next().unwrap();
        assert_eq!(stat, Stat::Max);
        assert_eq!(first.map(|f| i32::try_from(&f).unwrap()), exact(100));
        let snd = set.next().unwrap();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(snd.1.map(|s| i32::try_from(&s).unwrap()), exact(42));
    }

    #[test]
    fn merge_into_min() {
        let first = StatsSet::of(Stat::Min, exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Min).is_none());
    }

    #[test]
    fn merge_from_min() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Min, exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Min).is_none());
    }

    #[test]
    fn merge_mins() {
        let first = StatsSet::of(Stat::Min, exact(37)).merge_ordered(
            &StatsSet::of(Stat::Min, exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<i32>(Stat::Min), Some(exact(37)));
    }

    #[test]
    fn merge_into_max() {
        let first = StatsSet::of(Stat::Max, exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_none());
    }

    #[test]
    fn merge_from_max() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Max, exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_none());
    }

    #[test]
    fn merge_maxes() {
        let first = StatsSet::of(Stat::Max, exact(37)).merge_ordered(
            &StatsSet::of(Stat::Max, exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<i32>(Stat::Max), Some(exact(42)));
    }

    #[test]
    fn merge_into_scalar() {
        let first = StatsSet::of(Stat::TrueCount, exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::TrueCount).is_none());
    }

    #[test]
    fn merge_from_scalar() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::TrueCount, exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::TrueCount).is_none());
    }

    #[test]
    fn merge_scalars() {
        let first = StatsSet::of(Stat::TrueCount, exact(37)).merge_ordered(
            &StatsSet::of(Stat::TrueCount, exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<usize>(Stat::TrueCount), Some(exact(79usize)));
    }

    #[test]
    fn merge_into_freq() {
        let vec = (0usize..255).collect_vec();
        let first = StatsSet::of(Stat::BitWidthFreq, exact(vec)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::BitWidthFreq).is_none());
    }

    #[test]
    fn merge_from_freq() {
        let vec = (0usize..255).collect_vec();
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::BitWidthFreq, exact(vec)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::BitWidthFreq).is_none());
    }

    #[test]
    fn merge_freqs() {
        let vec_in = vec![5u64; 256];
        let vec_out = vec![10u64; 256];
        let first = StatsSet::of(Stat::BitWidthFreq, exact(vec_in.clone())).merge_ordered(
            &StatsSet::of(Stat::BitWidthFreq, exact(vec_in)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            first.get_as::<Vec<u64>>(Stat::BitWidthFreq),
            Some(exact(vec_out))
        );
    }

    #[test]
    fn merge_into_sortedness() {
        let first = StatsSet::of(Stat::IsStrictSorted, exact(true)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_from_sortedness() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::IsStrictSorted, exact(true)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_sortedness() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, exact(true));
        first.set(Stat::Max, exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, exact(true));
        second.set(Stat::Min, exact(2));
        first = first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            first.get_as::<bool>(Stat::IsStrictSorted),
            Some(exact(true))
        );
    }

    #[test]
    fn merge_sortedness_out_of_order() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, exact(true));
        first.set(Stat::Min, exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, exact(true));
        second.set(Stat::Max, exact(2));
        second = second.merge_ordered(
            &first,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            second.get_as::<bool>(Stat::IsStrictSorted),
            Some(exact(false))
        );
    }

    #[test]
    fn merge_sortedness_only_one_sorted() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, exact(true));
        first.set(Stat::Max, exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, exact(false));
        second.set(Stat::Min, exact(2));
        first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            second.get_as::<bool>(Stat::IsStrictSorted),
            Some(exact(false))
        );
    }

    #[test]
    fn merge_sortedness_missing_min() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, exact(true));
        first.set(Stat::Max, exact(1));
        let second = StatsSet::of(Stat::IsStrictSorted, exact(true));
        first = first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_unordered() {
        let array =
            PrimitiveArray::from_option_iter([Some(1), None, Some(2), Some(42), Some(10000), None])
                .into_array();
        let all_stats = all::<Stat>()
            .filter(|s| !matches!(s, Stat::TrueCount))
            .collect_vec();
        array.statistics().compute_all(&all_stats).unwrap();

        let stats = array.statistics().to_set();
        for stat in &all_stats {
            assert!(stats.get(*stat).is_some(), "Stat {} is missing", stat);
        }

        let merged = stats.clone().merge_unordered(
            &stats,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        for stat in &all_stats {
            assert_eq!(
                merged.get(*stat).is_some(),
                stat.is_commutative(),
                "Stat {} remains after merge_unordered despite not being commutative, or was removed despite being commutative",
                stat
            )
        }

        assert_eq!(
            merged.get_as::<i32>(Stat::Min),
            stats.get_as::<i32>(Stat::Min)
        );
        assert_eq!(
            merged.get_as::<i32>(Stat::Max),
            stats.get_as::<i32>(Stat::Max)
        );
        assert_eq!(
            merged.get_as::<u64>(Stat::NullCount).unwrap(),
            stats.get_as::<u64>(Stat::NullCount).unwrap().map(|s| s * 2)
        );
    }

    #[test]
    fn merge_min_bound_same() {
        // Merging a stat with a bound and another with an exact results in exact stat.
        // since bound for min is a lower bound, it can in fact contain any value >= bound.
        let merged = StatsSet::of(Stat::Min, bound(5)).merge_ordered(
            &StatsSet::of(Stat::Min, exact(5)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(merged.get_as::<i32>(Stat::Min), Some(exact(5)));
    }

    #[test]
    fn merge_min_bound_bound_lower() {
        let merged = StatsSet::of(Stat::Min, bound(4)).merge_ordered(
            &StatsSet::of(Stat::Min, exact(5)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(merged
            .get_as::<i32>(Stat::Min)
            .unwrap()
            .structural_eq(&bound(4)));
    }
}
