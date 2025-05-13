use std::fmt::Debug;

use enum_iterator::{Sequence, all};
use num_traits::CheckedAdd;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use super::traits::StatsProvider;
use super::{IsSorted, IsStrictSorted, NaNCount, NullCount, StatType, UncompressedSizeInBytes};
use crate::stats::{IsConstant, Max, Min, Precision, Stat, StatBound, StatsProviderExt, Sum};

#[derive(Default, Debug, Clone)]
pub struct StatsSet {
    values: Vec<(Stat, Precision<ScalarValue>)>,
}

impl StatsSet {
    /// Create new StatSet without validating uniqueness of all the entries
    ///
    /// # Safety
    ///
    /// This method will not panic or trigger UB, but may lead to duplicate stats being stored.
    pub fn new_unchecked(values: Vec<(Stat, Precision<ScalarValue>)>) -> Self {
        Self { values }
    }

    /// Specialized constructor for the case where the StatsSet represents
    /// an array consisting entirely of [null](vortex_dtype::DType::Null) values.
    pub fn nulls(len: usize) -> Self {
        let mut stats = Self::new_unchecked(vec![(Stat::NullCount, Precision::exact(len))]);

        if len > 0 {
            stats.set(Stat::IsConstant, Precision::exact(true));
            stats.set(Stat::IsSorted, Precision::exact(true));
            stats.set(Stat::IsStrictSorted, Precision::exact(len < 2));
        }

        stats
    }

    // A convenience method for creating a stats set which will represent an empty array.
    pub fn empty_array() -> StatsSet {
        StatsSet::new_unchecked(vec![(Stat::NullCount, Precision::exact(0))])
    }

    pub fn constant(scalar: Scalar, length: usize) -> Self {
        let (dtype, sv) = scalar.into_parts();
        let mut stats = Self::default();
        if length > 0 {
            stats.extend([
                (Stat::IsConstant, Precision::exact(true)),
                (Stat::IsSorted, Precision::exact(true)),
                (Stat::IsStrictSorted, Precision::exact(length <= 1)),
            ]);
        }

        let null_count = if sv.is_null() { length as u64 } else { 0 };
        stats.set(Stat::NullCount, Precision::exact(null_count));

        if !sv.is_null() {
            stats.extend([
                (Stat::Min, Precision::exact(sv.clone())),
                (Stat::Max, Precision::exact(sv.clone())),
            ]);
        }

        if matches!(dtype, DType::Bool(_)) {
            let bool_val = <Option<bool>>::try_from(&sv).vortex_expect("Checked dtype");
            let true_count = bool_val
                .map(|b| if b { length as u64 } else { 0 })
                .unwrap_or(0);
            stats.set(Stat::Sum, Precision::exact(true_count));
        }

        stats
    }

    pub fn bools_with_sum_and_null_count(true_count: usize, null_count: usize, len: usize) -> Self {
        StatsSet::new_unchecked(vec![
            (Stat::Sum, Precision::exact(true_count)),
            (Stat::NullCount, Precision::exact(null_count)),
            (Stat::Min, Precision::exact(true_count == len)),
            (Stat::Max, Precision::exact(true_count > 0)),
            (
                Stat::IsConstant,
                Precision::exact((true_count == 0 && null_count == 0) || true_count == len),
            ),
        ])
    }

    pub fn of(stat: Stat, value: Precision<ScalarValue>) -> Self {
        Self::new_unchecked(vec![(stat, value)])
    }

    fn reserve_full_capacity(&mut self) {
        if self.values.capacity() < Stat::CARDINALITY {
            self.values
                .reserve_exact(Stat::CARDINALITY - self.values.capacity());
        }
    }
}

// Getters and setters for individual stats.
impl StatsSet {
    /// Set the stat `stat` to `value`.
    pub fn set(&mut self, stat: Stat, value: Precision<ScalarValue>) {
        self.reserve_full_capacity();

        if let Some(existing) = self.values.iter_mut().find(|(s, _)| *s == stat) {
            *existing = (stat, value);
        } else {
            self.values.push((stat, value));
        }
    }

    /// Clear the stat `stat` from the set.
    pub fn clear(&mut self, stat: Stat) {
        self.values.retain(|(s, _)| *s != stat);
    }

    pub fn retain_only(&mut self, stats: &[Stat]) {
        self.values.retain(|(s, _)| stats.contains(s));
    }

    pub fn keep_inexact_stats(self, inexact_keep: &[Stat]) -> Self {
        self.values
            .into_iter()
            .filter_map(|(s, v)| inexact_keep.contains(&s).then(|| (s, v.into_inexact())))
            .collect()
    }

    /// Iterate over the statistic names and values in-place.
    ///
    /// See [Iterator].
    pub fn iter(&self) -> impl Iterator<Item = &(Stat, Precision<ScalarValue>)> {
        self.values.iter()
    }
}

// StatSetIntoIter just exists to protect current implementation from exposure on the public API.

/// Owned iterator over the stats.
///
/// See [IntoIterator].
pub struct StatsSetIntoIter(std::vec::IntoIter<(Stat, Precision<ScalarValue>)>);

impl Iterator for StatsSetIntoIter {
    type Item = (Stat, Precision<ScalarValue>);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl IntoIterator for StatsSet {
    type Item = (Stat, Precision<ScalarValue>);
    type IntoIter = StatsSetIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        StatsSetIntoIter(self.values.into_iter())
    }
}

impl FromIterator<(Stat, Precision<ScalarValue>)> for StatsSet {
    fn from_iter<T: IntoIterator<Item = (Stat, Precision<ScalarValue>)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut values = Vec::default();
        values.reserve_exact(Stat::CARDINALITY);

        let mut this = Self { values };
        this.extend(iter);
        this
    }
}

impl Extend<(Stat, Precision<ScalarValue>)> for StatsSet {
    #[inline]
    fn extend<T: IntoIterator<Item = (Stat, Precision<ScalarValue>)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve_full_capacity();

        iter.for_each(|(stat, value)| self.set(stat, value));
    }
}

// Merge helpers
impl StatsSet {
    /// Merge stats set `other` into `self`, with the semantic assumption that `other`
    /// contains stats from a disjoint array that is *appended* to the array represented by `self`.
    pub fn merge_ordered(mut self, other: &Self, dtype: &DType) -> Self {
        for s in all::<Stat>() {
            match s {
                Stat::IsConstant => self.merge_is_constant(other, dtype),
                Stat::IsSorted => self.merge_is_sorted(other, dtype),
                Stat::IsStrictSorted => self.merge_is_strict_sorted(other, dtype),
                Stat::Max => self.merge_max(other, dtype),
                Stat::Min => self.merge_min(other, dtype),
                Stat::Sum => self.merge_sum(other, dtype),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
                Stat::NaNCount => self.merge_nan_count(other),
            }
        }

        self
    }

    /// Merge stats set `other` into `self`, from a disjoint array, with no ordering assumptions.
    /// Stats that are not commutative (e.g., is_sorted) are dropped from the result.
    pub fn merge_unordered(mut self, other: &Self, dtype: &DType) -> Self {
        for s in all::<Stat>() {
            if !s.is_commutative() {
                self.clear(s);
                continue;
            }

            match s {
                Stat::IsConstant => self.merge_is_constant(other, dtype),
                Stat::Max => self.merge_max(other, dtype),
                Stat::Min => self.merge_min(other, dtype),
                Stat::Sum => self.merge_sum(other, dtype),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
                Stat::IsSorted | Stat::IsStrictSorted => {
                    unreachable!("not commutative")
                }
                Stat::NaNCount => self.merge_nan_count(other),
            }
        }

        self
    }

    // given two sets of stats (of differing precision) for the same array, combine them
    pub fn combine_sets(&mut self, other: &Self, dtype: &DType) -> VortexResult<()> {
        let other_stats: Vec<_> = other.values.iter().map(|(stat, _)| *stat).collect();
        for s in other_stats {
            match s {
                Stat::Max => self.combine_bound::<Max>(other, dtype)?,
                Stat::Min => self.combine_bound::<Min>(other, dtype)?,
                Stat::UncompressedSizeInBytes => {
                    self.combine_bound::<UncompressedSizeInBytes>(other, dtype)?
                }
                Stat::IsConstant => self.combine_bool_stat::<IsConstant>(other)?,
                Stat::IsSorted => self.combine_bool_stat::<IsSorted>(other)?,
                Stat::IsStrictSorted => self.combine_bool_stat::<IsStrictSorted>(other)?,
                Stat::NullCount => self.combine_bound::<NullCount>(other, dtype)?,
                Stat::Sum => self.combine_bound::<Sum>(other, dtype)?,
                Stat::NaNCount => self.combine_bound::<NaNCount>(other, dtype)?,
            }
        }
        Ok(())
    }

    fn combine_bound<S: StatType<Scalar>>(
        &mut self,
        other: &Self,
        dtype: &DType,
    ) -> VortexResult<()>
    where
        S::Bound: StatBound<Scalar> + Debug + Eq + PartialEq,
    {
        match (
            self.get_scalar_bound::<S>(dtype),
            other.get_scalar_bound::<S>(dtype),
        ) {
            (Some(m1), Some(m2)) => {
                let meet = m1
                    .intersection(&m2)
                    .vortex_expect("can always compare scalar")
                    .ok_or_else(|| {
                        vortex_err!("{:?} bounds ({m1:?}, {m2:?}) do not overlap", S::STAT)
                    })?;
                if meet != m1 {
                    self.set(S::STAT, meet.into_value().map(Scalar::into_value));
                }
            }
            (None, Some(m)) => self.set(S::STAT, m.into_value().map(Scalar::into_value)),
            (Some(_), _) => (),
            (None, None) => self.clear(S::STAT),
        }
        Ok(())
    }

    fn combine_bool_stat<S: StatType<bool>>(&mut self, other: &Self) -> VortexResult<()>
    where
        S::Bound: StatBound<bool> + Debug + Eq + PartialEq,
    {
        match (
            self.get_as_bound::<S, bool>(),
            other.get_as_bound::<S, bool>(),
        ) {
            (Some(m1), Some(m2)) => {
                let intersection = m1
                    .intersection(&m2)
                    .vortex_expect("can always compare boolean")
                    .ok_or_else(|| {
                        vortex_err!("{:?} bounds ({m1:?}, {m2:?}) do not overlap", S::STAT)
                    })?;
                if intersection != m1 {
                    self.set(S::STAT, intersection.into_value().map(ScalarValue::from));
                }
            }
            (None, Some(m)) => self.set(S::STAT, m.into_value().map(ScalarValue::from)),
            (Some(_), None) => (),
            (None, None) => self.clear(S::STAT),
        }
        Ok(())
    }

    fn merge_min(&mut self, other: &Self, dtype: &DType) {
        match (
            self.get_scalar_bound::<Min>(dtype),
            other.get_scalar_bound::<Min>(dtype),
        ) {
            (Some(m1), Some(m2)) => {
                let meet = m1.union(&m2).vortex_expect("can compare scalar");
                if meet != m1 {
                    self.set(Stat::Min, meet.into_value().map(Scalar::into_value));
                }
            }
            _ => self.clear(Stat::Min),
        }
    }

    fn merge_max(&mut self, other: &Self, dtype: &DType) {
        match (
            self.get_scalar_bound::<Max>(dtype),
            other.get_scalar_bound::<Max>(dtype),
        ) {
            (Some(m1), Some(m2)) => {
                let meet = m1.union(&m2).vortex_expect("can compare scalar");
                if meet != m1 {
                    self.set(Stat::Max, meet.into_value().map(Scalar::into_value));
                }
            }
            _ => self.clear(Stat::Max),
        }
    }

    fn merge_sum(&mut self, other: &Self, dtype: &DType) {
        match (
            self.get_scalar_bound::<Sum>(dtype),
            other.get_scalar_bound::<Sum>(dtype),
        ) {
            (Some(m1), Some(m2)) => {
                // If the combine sum is exact, then we can sum them.
                if let Some(scalar_value) = m1.zip(m2).as_exact().and_then(|(s1, s2)| {
                    s1.as_primitive()
                        .checked_add(&s2.as_primitive())
                        .map(|pscalar| {
                            pscalar
                                .pvalue()
                                .map(|pvalue| {
                                    Scalar::primitive_value(
                                        pvalue,
                                        pscalar.ptype(),
                                        pscalar.dtype().nullability(),
                                    )
                                    .into_value()
                                })
                                .unwrap_or_else(ScalarValue::null)
                        })
                }) {
                    self.set(Stat::Sum, Precision::Exact(scalar_value));
                }
            }
            _ => self.clear(Stat::Sum),
        }
    }

    fn merge_is_constant(&mut self, other: &Self, dtype: &DType) {
        let self_const = self.get_as(Stat::IsConstant);
        let other_const = other.get_as(Stat::IsConstant);
        let self_min = self.get_scalar(Stat::Min, dtype);
        let other_min = other.get_scalar(Stat::Min, dtype);

        if let (
            Some(Precision::Exact(self_const)),
            Some(Precision::Exact(other_const)),
            Some(Precision::Exact(self_min)),
            Some(Precision::Exact(other_min)),
        ) = (self_const, other_const, self_min, other_min)
        {
            if self_const && other_const && self_min == other_min {
                self.set(Stat::IsConstant, Precision::exact(true));
            } else {
                self.set(Stat::IsConstant, Precision::inexact(false));
            }
        }
        self.set(Stat::IsConstant, Precision::exact(false));
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
            if let (Some(self_max), Some(other_min)) = (
                self.get_scalar_bound::<Max>(dtype),
                other.get_scalar_bound::<Min>(dtype),
            ) {
                return if cmp(&self_max.max_value(), &other_min.min_value()) {
                    // keep value
                } else {
                    self.set(stat, Precision::inexact(false));
                };
            }
        }
        self.clear(stat);
    }

    fn merge_null_count(&mut self, other: &Self) {
        self.merge_sum_stat(Stat::NullCount, other)
    }

    fn merge_nan_count(&mut self, other: &Self) {
        self.merge_sum_stat(Stat::NaNCount, other)
    }

    fn merge_uncompressed_size_in_bytes(&mut self, other: &Self) {
        self.merge_sum_stat(Stat::UncompressedSizeInBytes, other)
    }

    fn merge_sum_stat(&mut self, stat: Stat, other: &Self) {
        match (self.get_as::<usize>(stat), other.get_as::<usize>(stat)) {
            (Some(nc1), Some(nc2)) => {
                self.set(
                    stat,
                    nc1.zip(nc2).map(|(nc1, nc2)| ScalarValue::from(nc1 + nc2)),
                );
            }
            _ => self.clear(stat),
        }
    }
}

impl StatsProvider for StatsSet {
    fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        self.values
            .iter()
            .find(|(s, _)| *s == stat)
            .map(|(_, v)| v.clone())
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;
    use itertools::Itertools;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::arrays::PrimitiveArray;
    use crate::stats::{IsConstant, Precision, Stat, StatsProvider, StatsProviderExt, StatsSet};

    #[test]
    fn test_iter() {
        let set = StatsSet::new_unchecked(vec![
            (Stat::Max, Precision::exact(100)),
            (Stat::Min, Precision::exact(42)),
        ]);
        let mut iter = set.iter();
        let first = iter.next().unwrap().clone();
        assert_eq!(first.0, Stat::Max);
        assert_eq!(
            first.1.map(|f| i32::try_from(&f).unwrap()),
            Precision::exact(100)
        );
        let snd = iter.next().unwrap().clone();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(snd.1.map(|s| i32::try_from(&s).unwrap()), 42);
    }

    #[test]
    fn into_iter() {
        let mut set = StatsSet::new_unchecked(vec![
            (Stat::Max, Precision::exact(100)),
            (Stat::Min, Precision::exact(42)),
        ])
        .into_iter();
        let (stat, first) = set.next().unwrap();
        assert_eq!(stat, Stat::Max);
        assert_eq!(
            first.map(|f| i32::try_from(&f).unwrap()),
            Precision::exact(100)
        );
        let snd = set.next().unwrap();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(
            snd.1.map(|s| i32::try_from(&s).unwrap()),
            Precision::exact(42)
        );
    }

    #[test]
    fn merge_constant() {
        let first = StatsSet::from_iter([
            (Stat::Min, Precision::exact(42)),
            (Stat::IsConstant, Precision::exact(true)),
        ])
        .merge_ordered(
            &StatsSet::from_iter([
                (Stat::Min, Precision::inexact(42)),
                (Stat::IsConstant, Precision::exact(true)),
            ]),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            first.get_as::<bool>(Stat::IsConstant),
            Some(Precision::exact(false))
        );
        assert_eq!(first.get_as::<i32>(Stat::Min), Some(Precision::exact(42)));
    }

    #[test]
    fn merge_into_min() {
        let first = StatsSet::of(Stat::Min, Precision::exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Min).is_none());
    }

    #[test]
    fn merge_from_min() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Min).is_none());
    }

    #[test]
    fn merge_mins() {
        let first = StatsSet::of(Stat::Min, Precision::exact(37)).merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<i32>(Stat::Min), Some(Precision::exact(37)));
    }

    #[test]
    fn merge_into_bound_max() {
        let first = StatsSet::of(Stat::Max, Precision::exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_none());
    }

    #[test]
    fn merge_from_max() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Max, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_none());
    }

    #[test]
    fn merge_maxes() {
        let first = StatsSet::of(Stat::Max, Precision::exact(37)).merge_ordered(
            &StatsSet::of(Stat::Max, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(first.get_as::<i32>(Stat::Max), Some(Precision::exact(42)));
    }

    #[test]
    fn merge_maxes_bound() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let first = StatsSet::of(Stat::Max, Precision::exact(42i32))
            .merge_ordered(&StatsSet::of(Stat::Max, Precision::inexact(43i32)), &dtype);
        assert_eq!(first.get_as::<i32>(Stat::Max), Some(Precision::inexact(43)));
    }

    #[test]
    fn merge_into_scalar() {
        let first = StatsSet::of(Stat::Sum, Precision::exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Sum).is_none());
    }

    #[test]
    fn merge_from_scalar() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Sum, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Sum).is_none());
    }

    #[test]
    fn merge_scalars() {
        let first = StatsSet::of(Stat::Sum, Precision::exact(37)).merge_ordered(
            &StatsSet::of(Stat::Sum, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            first.get_as::<usize>(Stat::Sum),
            Some(Precision::exact(79usize))
        );
    }

    #[test]
    fn merge_into_sortedness() {
        let first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_from_sortedness() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::IsStrictSorted, Precision::exact(true)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_sortedness() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        first.set(Stat::Max, Precision::exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        second.set(Stat::Min, Precision::exact(2));
        first = first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            first.get_as::<bool>(Stat::IsStrictSorted),
            Some(Precision::exact(true))
        );
    }

    #[test]
    fn merge_sortedness_out_of_order() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        first.set(Stat::Min, Precision::exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        second.set(Stat::Max, Precision::exact(2));
        second = second.merge_ordered(
            &first,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            second.get_as::<bool>(Stat::IsStrictSorted),
            Some(Precision::inexact(false))
        );
    }

    #[test]
    fn merge_sortedness_only_one_sorted() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        first.set(Stat::Max, Precision::exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, Precision::exact(false));
        second.set(Stat::Min, Precision::exact(2));
        first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            second.get_as::<bool>(Stat::IsStrictSorted),
            Some(Precision::exact(false))
        );
    }

    #[test]
    fn merge_sortedness_missing_min() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        first.set(Stat::Max, Precision::exact(1));
        let second = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        first = first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_none());
    }

    #[test]
    fn merge_sortedness_bound_min() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        first.set(Stat::Max, Precision::exact(1));
        let mut second = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true));
        second.set(Stat::Min, Precision::inexact(2));
        first = first.merge_ordered(
            &second,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(
            first.get_as::<bool>(Stat::IsStrictSorted),
            Some(Precision::exact(true))
        );
    }

    #[test]
    fn merge_unordered() {
        let array =
            PrimitiveArray::from_option_iter([Some(1), None, Some(2), Some(42), Some(10000), None]);
        let all_stats = all::<Stat>()
            .filter(|s| !matches!(s, Stat::Sum))
            .filter(|s| !matches!(s, Stat::NaNCount))
            .collect_vec();
        array.statistics().compute_all(&all_stats).unwrap();

        let stats = array.statistics().to_owned();
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
        let merged = StatsSet::of(Stat::Min, Precision::inexact(5)).merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(5)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(merged.get_as::<i32>(Stat::Min), Some(Precision::exact(5)));
    }

    #[test]
    fn merge_min_bound_bound_lower() {
        let merged = StatsSet::of(Stat::Min, Precision::inexact(4)).merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(5)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert_eq!(merged.get_as::<i32>(Stat::Min), Some(Precision::inexact(4)));
    }

    #[test]
    fn retain_approx() {
        let set = StatsSet::from_iter([
            (Stat::Max, Precision::exact(100)),
            (Stat::Min, Precision::exact(50)),
            (Stat::Sum, Precision::inexact(10)),
        ]);

        let set = set.keep_inexact_stats(&[Stat::Min, Stat::Max]);

        assert_eq!(set.len(), 2);
        assert_eq!(set.get_as::<i32>(Stat::Max), Some(Precision::inexact(100)));
        assert_eq!(set.get_as::<i32>(Stat::Min), Some(Precision::inexact(50)));
        assert_eq!(set.get_as::<i32>(Stat::Sum), None);
    }

    #[test]
    fn test_combine_is_constant() {
        {
            let mut stats = StatsSet::of(Stat::IsConstant, Precision::exact(true));
            let stats2 = StatsSet::of(Stat::IsConstant, Precision::exact(true));
            stats.combine_bool_stat::<IsConstant>(&stats2).unwrap();
            assert_eq!(
                stats.get_as::<bool>(Stat::IsConstant),
                Some(Precision::exact(true))
            );
        }

        {
            let mut stats = StatsSet::of(Stat::IsConstant, Precision::exact(true));
            let stats2 = StatsSet::of(Stat::IsConstant, Precision::inexact(false));
            stats.combine_bool_stat::<IsConstant>(&stats2).unwrap();
            assert_eq!(
                stats.get_as::<bool>(Stat::IsConstant),
                Some(Precision::exact(true))
            );
        }

        {
            let mut stats = StatsSet::of(Stat::IsConstant, Precision::exact(false));
            let stats2 = StatsSet::of(Stat::IsConstant, Precision::inexact(false));
            stats.combine_bool_stat::<IsConstant>(&stats2).unwrap();
            assert_eq!(
                stats.get_as::<bool>(Stat::IsConstant),
                Some(Precision::exact(false))
            );
        }
    }

    #[test]
    fn test_combine_sets_boolean_conflict() {
        let mut stats1 = StatsSet::from_iter([
            (Stat::IsConstant, Precision::exact(true)),
            (Stat::IsSorted, Precision::exact(true)),
        ]);

        let stats2 = StatsSet::from_iter([
            (Stat::IsConstant, Precision::exact(false)),
            (Stat::IsSorted, Precision::exact(true)),
        ]);

        let result = stats1.combine_sets(
            &stats2,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_combine_sets_with_missing_stats() {
        let mut stats1 = StatsSet::from_iter([
            (Stat::Min, Precision::exact(42)),
            (Stat::UncompressedSizeInBytes, Precision::exact(1000)),
        ]);

        let stats2 = StatsSet::from_iter([
            (Stat::Max, Precision::exact(100)),
            (Stat::IsStrictSorted, Precision::exact(true)),
        ]);

        stats1
            .combine_sets(
                &stats2,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
            )
            .unwrap();

        // Min should remain unchanged
        assert_eq!(stats1.get_as::<i32>(Stat::Min), Some(Precision::exact(42)));
        // Max should be added
        assert_eq!(stats1.get_as::<i32>(Stat::Max), Some(Precision::exact(100)));
        // IsStrictSorted should be added
        assert_eq!(
            stats1.get_as::<bool>(Stat::IsStrictSorted),
            Some(Precision::exact(true))
        );
    }

    #[test]
    fn test_combine_sets_with_inexact() {
        let mut stats1 = StatsSet::from_iter([
            (Stat::Min, Precision::exact(42)),
            (Stat::Max, Precision::inexact(100)),
            (Stat::IsConstant, Precision::exact(false)),
        ]);

        let stats2 = StatsSet::from_iter([
            // Must ensure Min from stats2 is <= Min from stats1
            (Stat::Min, Precision::inexact(40)),
            (Stat::Max, Precision::exact(90)),
            (Stat::IsSorted, Precision::exact(true)),
        ]);

        stats1
            .combine_sets(
                &stats2,
                &DType::Primitive(PType::I32, Nullability::NonNullable),
            )
            .unwrap();

        // Min should remain unchanged since it's more restrictive than the inexact value
        assert_eq!(stats1.get_as::<i32>(Stat::Min), Some(Precision::exact(42)));
        // Check that max was updated with the exact value
        assert_eq!(stats1.get_as::<i32>(Stat::Max), Some(Precision::exact(90)));
        // Check that IsSorted was added
        assert_eq!(
            stats1.get_as::<bool>(Stat::IsSorted),
            Some(Precision::exact(true))
        );
    }
}
