use enum_iterator::{all, Sequence};
use itertools::{EitherOrBoth, Itertools};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexError, VortexExpect, VortexUnwrap};
use vortex_scalar::{Scalar, ScalarValue};

use crate::stats::Stat;

#[derive(Default, Debug, Clone)]
pub struct StatsSet {
    values: Option<Vec<(Stat, ScalarValue)>>,
}

impl StatsSet {
    /// Create new StatSet without validating uniqueness of all the entries
    ///
    /// # Safety
    ///
    /// This method will not panic or trigger UB, but may lead to duplicate stats being stored.
    pub fn new_unchecked(values: Vec<(Stat, ScalarValue)>) -> Self {
        Self {
            values: Some(values),
        }
    }

    /// Specialized constructor for the case where the StatsSet represents
    /// an array consisting entirely of [null](vortex_dtype::DType::Null) values.
    pub fn nulls(len: usize, dtype: &DType) -> Self {
        let mut stats = Self::new_unchecked(vec![
            (Stat::RunCount, 1.into()),
            (Stat::NullCount, len.into()),
        ]);

        if len > 0 {
            stats.set(Stat::IsConstant, true);
            stats.set(Stat::IsSorted, true);
            stats.set(Stat::IsStrictSorted, len < 2);
        }

        // Add any DType-specific stats.
        match dtype {
            DType::Bool(_) => {
                stats.set(Stat::TrueCount, 0);
            }
            DType::Primitive(ptype, _) => {
                ptype.byte_width();
                stats.set(Stat::BitWidthFreq, vec![0u64; ptype.byte_width() * 8 + 1]);
                stats.set(
                    Stat::TrailingZeroFreq,
                    vec![ptype.byte_width() as u64 * 8; ptype.byte_width() * 8 + 1],
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
            stats.set(Stat::IsConstant, true);
            stats.set(Stat::IsSorted, true);
            stats.set(Stat::IsStrictSorted, length <= 1);
        }

        let run_count = if length == 0 { 0u64 } else { 1 };
        stats.set(Stat::RunCount, run_count);

        let null_count = if sv.is_null() { length as u64 } else { 0 };
        stats.set(Stat::NullCount, null_count);

        if !sv.is_null() {
            stats.set(Stat::Min, sv.clone());
            stats.set(Stat::Max, sv.clone());
        }

        if matches!(dtype, DType::Bool(_)) {
            let bool_val = <Option<bool>>::try_from(&sv).vortex_expect("Checked dtype");
            let true_count = bool_val
                .map(|b| if b { length as u64 } else { 0 })
                .unwrap_or(0);
            stats.set(Stat::TrueCount, true_count);
        }

        stats
    }

    pub fn bools_with_true_and_null_count(
        true_count: usize,
        null_count: usize,
        len: usize,
    ) -> Self {
        StatsSet::new_unchecked(vec![
            (Stat::TrueCount, true_count.into()),
            (Stat::NullCount, null_count.into()),
            (Stat::Min, (true_count == len).into()),
            (Stat::Max, (true_count > 0).into()),
            (
                Stat::IsConstant,
                ((true_count == 0 && null_count == 0) || true_count == len).into(),
            ),
        ])
    }

    pub fn of<S: Into<ScalarValue>>(stat: Stat, value: S) -> Self {
        Self::new_unchecked(vec![(stat, value.into())])
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

    pub fn get(&self, stat: Stat) -> Option<&ScalarValue> {
        self.values
            .as_ref()
            .and_then(|v| v.iter().find(|(s, _)| *s == stat).map(|(_, v)| v))
    }

    pub fn get_as<T: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<T> {
        self.get(stat).map(|v| {
            T::try_from(v).unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to get stat {} as {}",
                    stat,
                    std::any::type_name::<T>()
                )
            })
        })
    }

    /// Set the stat `stat` to `value`.
    pub fn set<S: Into<ScalarValue>>(&mut self, stat: Stat, value: S) {
        if self.values.is_none() {
            self.values = Some(Vec::with_capacity(Stat::CARDINALITY));
        }
        let values = self.values.as_mut().vortex_expect("we just initialized it");
        if let Some(existing) = values.iter_mut().find(|(s, _)| *s == stat) {
            *existing = (stat, value.into());
        } else {
            values.push((stat, value.into()));
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
    pub fn iter(&self) -> impl Iterator<Item = &(Stat, ScalarValue)> {
        self.values.iter().flat_map(|v| v.iter())
    }
}

// StatSetIntoIter just exists to protect current implementation from exposure on the public API.

/// Owned iterator over the stats.
///
/// See [IntoIterator].
pub struct StatsSetIntoIter(Option<std::vec::IntoIter<(Stat, ScalarValue)>>);

impl Iterator for StatsSetIntoIter {
    type Item = (Stat, ScalarValue);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(|i| i.next())
    }
}

impl IntoIterator for StatsSet {
    type Item = (Stat, ScalarValue);
    type IntoIter = StatsSetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        StatsSetIntoIter(self.values.map(|v| v.into_iter()))
    }
}

impl FromIterator<(Stat, ScalarValue)> for StatsSet {
    fn from_iter<T: IntoIterator<Item = (Stat, ScalarValue)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut this = Self {
            values: Some(Vec::with_capacity(lower_bound)),
        };
        this.extend(iter);
        this
    }
}

impl Extend<(Stat, ScalarValue)> for StatsSet {
    #[inline]
    fn extend<T: IntoIterator<Item = (Stat, ScalarValue)>>(&mut self, iter: T) {
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
        match (self.get(Stat::Min), other.get(Stat::Min)) {
            (Some(m1), Some(m2)) => {
                if Scalar::new(dtype.clone(), m2.clone()) < Scalar::new(dtype.clone(), m1.clone()) {
                    self.set(Stat::Min, m2.clone());
                }
            }
            _ => self.clear(Stat::Min),
        }
    }

    fn merge_max(&mut self, other: &Self, dtype: &DType) {
        match (self.get(Stat::Max), other.get(Stat::Max)) {
            (Some(m1), Some(m2)) => {
                if Scalar::new(dtype.clone(), m2.clone()) > Scalar::new(dtype.clone(), m1.clone()) {
                    self.set(Stat::Max, m2.clone());
                }
            }
            _ => self.clear(Stat::Max),
        }
    }

    fn merge_is_constant(&mut self, other: &Self, dtype: &DType) {
        if let Some(is_constant) = self.get_as(Stat::IsConstant) {
            if let Some(other_is_constant) = other.get_as(Stat::IsConstant) {
                if is_constant
                    && other_is_constant
                    && self
                        .get(Stat::Min)
                        .cloned()
                        .map(|sv| Scalar::new(dtype.clone(), sv))
                        == other
                            .get(Stat::Min)
                            .cloned()
                            .map(|sv| Scalar::new(dtype.clone(), sv))
                {
                    return;
                }
            }
            self.set(Stat::IsConstant, false);
        }
    }

    fn merge_is_sorted(&mut self, other: &Self, dtype: &DType) {
        self.merge_sortedness_stat(other, Stat::IsSorted, dtype, |own, other| own <= other)
    }

    fn merge_is_strict_sorted(&mut self, other: &Self, dtype: &DType) {
        self.merge_sortedness_stat(other, Stat::IsStrictSorted, dtype, |own, other| own < other)
    }

    fn merge_sortedness_stat<F: Fn(Option<Scalar>, Option<Scalar>) -> bool>(
        &mut self,
        other: &Self,
        stat: Stat,
        dtype: &DType,
        cmp: F,
    ) {
        if let Some(is_sorted) = self.get_as(stat) {
            if let Some(other_is_sorted) = other.get_as(stat) {
                if !(self.get(Stat::Max).is_some() && other.get(Stat::Min).is_some()) {
                    self.clear(stat);
                } else if is_sorted
                    && other_is_sorted
                    && cmp(
                        self.get(Stat::Max)
                            .cloned()
                            .map(|sv| Scalar::new(dtype.clone(), sv)),
                        other
                            .get(Stat::Min)
                            .cloned()
                            .map(|sv| Scalar::new(dtype.clone(), sv)),
                    )
                {
                    return;
                } else {
                    self.set(stat, false);
                }
            } else {
                self.clear(stat)
            }
        }
    }

    fn merge_true_count(&mut self, other: &Self) {
        self.merge_sum_stat(other, Stat::TrueCount)
    }

    fn merge_null_count(&mut self, other: &Self) {
        self.merge_sum_stat(other, Stat::NullCount)
    }

    fn merge_uncompressed_size_in_bytes(&mut self, other: &Self) {
        self.merge_sum_stat(other, Stat::UncompressedSizeInBytes)
    }

    fn merge_sum_stat(&mut self, other: &Self, stat: Stat) {
        match (self.get_as::<usize>(stat), other.get_as::<usize>(stat)) {
            (Some(nc1), Some(nc2)) => {
                self.set(stat, nc1 + nc2);
            }
            _ => self.clear(stat),
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
                let combined_freq = f1
                    .iter()
                    .zip_longest(f2.iter())
                    .map(|pair| match pair {
                        EitherOrBoth::Both(a, b) => a + b,
                        EitherOrBoth::Left(a) => *a,
                        EitherOrBoth::Right(b) => *b,
                    })
                    .collect_vec();
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
                self.set(Stat::RunCount, r1 + r2 + 1);
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
    use crate::stats::{ArrayStatistics as _, Stat, StatsSet};
    use crate::IntoArrayData as _;

    #[test]
    fn test_iter() {
        let set = StatsSet::new_unchecked(vec![(Stat::Max, 100.into()), (Stat::Min, 42.into())]);
        let mut iter = set.iter();
        let first = iter.next().unwrap();
        assert_eq!(first.0, Stat::Max);
        assert_eq!(i32::try_from(&first.1).unwrap(), 100);
        let snd = iter.next().unwrap();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(i32::try_from(&snd.1).unwrap(), 42);
    }

    #[test]
    fn into_iter() {
        let mut set =
            StatsSet::new_unchecked(vec![(Stat::Max, 100.into()), (Stat::Min, 42.into())])
                .into_iter();
        let first = set.next().unwrap();
        assert_eq!(first.0, Stat::Max);
        assert_eq!(i32::try_from(&first.1).unwrap(), 100);
        let snd = set.next().unwrap();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(i32::try_from(&snd.1).unwrap(), 42);
    }

    #[test]
    fn merge_into_min() {
        let first = StatsSet::of(Stat::Min, 42).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Min).is_none());
    }

    #[test]
    fn merge_from_min() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Min, 42),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Min).is_none());
    }

    #[test]
    fn merge_mins() {
        let first = StatsSet::of(Stat::Min, 37).merge_ordered(
            &StatsSet::of(Stat::Min, 42),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<i32>(Stat::Min), Some(37));
    }

    #[test]
    fn merge_into_max() {
        let first = StatsSet::of(Stat::Max, 42).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_none());
    }

    #[test]
    fn merge_from_max() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Max, 42),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_none());
    }

    #[test]
    fn merge_maxes() {
        let first = StatsSet::of(Stat::Max, 37).merge_ordered(
            &StatsSet::of(Stat::Max, 42),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<i32>(Stat::Max), Some(42));
    }

    #[test]
    fn merge_into_scalar() {
        let first = StatsSet::of(Stat::TrueCount, 42).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::TrueCount).is_none());
    }

    #[test]
    fn merge_from_scalar() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::TrueCount, 42),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::TrueCount).is_none());
    }

    #[test]
    fn merge_scalars() {
        let first = StatsSet::of(Stat::TrueCount, 37).merge_ordered(
            &StatsSet::of(Stat::TrueCount, 42),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<usize>(Stat::TrueCount), Some(79));
    }

    #[test]
    fn merge_into_freq() {
        let vec = (0usize..255).collect_vec();
        let first = StatsSet::of(Stat::BitWidthFreq, vec).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::BitWidthFreq).is_none());
    }

    #[test]
    fn merge_from_freq() {
        let vec = (0usize..255).collect_vec();
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::BitWidthFreq, vec),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::BitWidthFreq).is_none());
    }

    #[test]
    fn merge_freqs() {
        let vec_in = vec![5u64; 256];
        let vec_out = vec![10u64; 256];
        let first = StatsSet::of(Stat::BitWidthFreq, vec_in.clone()).merge_ordered(
            &StatsSet::of(Stat::BitWidthFreq, vec_in),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<Vec<u64>>(Stat::BitWidthFreq), Some(vec_out));
    }

    #[test]
    fn merge_into_sortedness() {
        let first = StatsSet::of(Stat::IsStrictSorted, true).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_from_sortedness() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::IsStrictSorted, true),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_sortedness() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1);
        let mut second = StatsSet::of(Stat::IsStrictSorted, true);
        second.set(Stat::Min, 2);
        first = first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<bool>(Stat::IsStrictSorted), Some(true));
    }

    #[test]
    fn merge_sortedness_out_of_order() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Min, 1);
        let mut second = StatsSet::of(Stat::IsStrictSorted, true);
        second.set(Stat::Max, 2);
        second = second.merge_ordered(
            &first,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(second.get_as::<bool>(Stat::IsStrictSorted), Some(false));
    }

    #[test]
    fn merge_sortedness_only_one_sorted() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1);
        let mut second = StatsSet::of(Stat::IsStrictSorted, false);
        second.set(Stat::Min, 2);
        first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(second.get_as::<bool>(Stat::IsStrictSorted), Some(false));
    }

    #[test]
    fn merge_sortedness_missing_min() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1);
        let second = StatsSet::of(Stat::IsStrictSorted, true);
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
            2 * stats.get_as::<u64>(Stat::NullCount).unwrap()
        );
    }
}
