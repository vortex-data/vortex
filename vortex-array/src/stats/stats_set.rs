// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use enum_iterator::all;
use num_traits::CheckedAdd;
use smallvec::SmallVec;
use smallvec::smallvec;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::expr::stats::IsConstant;
use crate::expr::stats::IsSorted;
use crate::expr::stats::IsStrictSorted;
use crate::expr::stats::Max;
use crate::expr::stats::Min;
use crate::expr::stats::NaNCount;
use crate::expr::stats::NullCount;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatBound;
use crate::expr::stats::StatType;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::expr::stats::Sum;
use crate::expr::stats::UncompressedSizeInBytes;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// Type of the SmallVec stored inside StatsSet
pub type StatsArray = [(Stat, Precision<ScalarValue>); 4];

#[derive(Default, Debug, Clone)]
pub struct StatsSet {
    values: SmallVec<StatsArray>,
}

impl StatsSet {
    /// Create new StatSet without validating uniqueness of all the entries
    ///
    /// # Safety
    ///
    /// This method will not panic or trigger UB, but may lead to duplicate stats being stored.
    pub unsafe fn new_unchecked(values: SmallVec<StatsArray>) -> Self {
        Self { values }
    }

    /// Create StatsSet from single stat and value
    pub fn of(stat: Stat, value: Precision<ScalarValue>) -> Self {
        Self {
            values: smallvec![(stat, value)],
        }
    }

    /// Wrap stats set with a dtype for mutable typed scalar access
    pub fn as_mut_typed_ref<'a, 'b>(&'a mut self, dtype: &'b DType) -> MutTypedStatsSetRef<'a, 'b> {
        MutTypedStatsSetRef {
            values: self,
            dtype,
        }
    }

    /// Wrap stats set with a dtype for typed scalar access
    pub fn as_typed_ref<'a, 'b>(&'a self, dtype: &'b DType) -> TypedStatsSetRef<'a, 'b> {
        TypedStatsSetRef {
            values: self,
            dtype,
        }
    }
}

// Getters and setters for individual stats.
impl StatsSet {
    /// Set the stat `stat` to `value`.
    pub fn set(&mut self, stat: Stat, value: Precision<ScalarValue>) {
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

    /// Only keep given stats
    pub fn retain_only(&mut self, stats: &[Stat]) {
        self.values.retain(|(s, _)| stats.contains(s));
    }

    /// Iterate over the statistic names and values in-place.
    ///
    /// See [Iterator].
    pub fn iter(&self) -> impl Iterator<Item = &(Stat, Precision<ScalarValue>)> {
        self.values.iter()
    }

    /// Get value for a given stat
    pub fn get(&self, stat: Stat) -> Precision<ScalarValue> {
        self.values
            .iter()
            .find(|(s, _)| *s == stat)
            .map(|(_, v)| v.clone())
            .unwrap_or(Precision::Absent)
    }

    /// Length of the stats set
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check whether the statset is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get scalar value of a given dtype
    pub fn get_as<T: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
        dtype: &DType,
    ) -> Precision<T> {
        self.get(stat).map(|v| {
            T::try_from(
                &Scalar::try_new(dtype.clone(), Some(v))
                    .vortex_expect("failed to construct a scalar statistic"),
            )
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to get stat {} as {}",
                    stat,
                    std::any::type_name::<T>()
                )
            })
        })
    }
}

// StatSetIntoIter just exists to protect current implementation from exposure on the public API.

/// Owned iterator over the stats.
///
/// See [IntoIterator].
pub struct StatsSetIntoIter(smallvec::IntoIter<StatsArray>);

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

        let mut this = Self {
            values: SmallVec::new(),
        };
        this.extend(iter);
        this
    }
}

impl Extend<(Stat, Precision<ScalarValue>)> for StatsSet {
    #[inline]
    fn extend<T: IntoIterator<Item = (Stat, Precision<ScalarValue>)>>(&mut self, iter: T) {
        iter.into_iter()
            .for_each(|(stat, value)| self.set(stat, value));
    }
}

/// Merge helpers
impl StatsSet {
    /// Merge stats set `other` into `self`, with the semantic assumption that `other`
    /// contains stats from a disjoint array that is *appended* to the array represented by `self`.
    pub fn merge_ordered(mut self, other: &Self, dtype: &DType) -> Self {
        self.as_mut_typed_ref(dtype)
            .merge_ordered(&other.as_typed_ref(dtype));
        self
    }

    /// Merge stats set `other` into `self`, from a disjoint array, with no ordering assumptions.
    /// Stats that are not commutative (e.g., is_sorted) are dropped from the result.
    pub fn merge_unordered(mut self, other: &Self, dtype: &DType) -> Self {
        self.as_mut_typed_ref(dtype)
            .merge_unordered(&other.as_typed_ref(dtype));
        self
    }

    /// Given two sets of stats (of differing precision) for the same array, combine them
    pub fn combine_sets(&mut self, other: &Self, dtype: &DType) -> VortexResult<()> {
        self.as_mut_typed_ref(dtype)
            .combine_sets(&other.as_typed_ref(dtype))
    }
}

pub struct TypedStatsSetRef<'a, 'b> {
    pub values: &'a StatsSet,
    pub dtype: &'b DType,
}

impl StatsProvider for TypedStatsSetRef<'_, '_> {
    fn get(&self, stat: Stat) -> Precision<Scalar> {
        self.values.get(stat).map(|sv| {
            Scalar::try_new(
                stat.dtype(self.dtype)
                    .vortex_expect("Must have valid dtype if value is present"),
                Some(sv),
            )
            .vortex_expect("failed to construct a scalar statistic")
        })
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

pub struct MutTypedStatsSetRef<'a, 'b> {
    pub values: &'a mut StatsSet,
    pub dtype: &'b DType,
}

impl MutTypedStatsSetRef<'_, '_> {
    /// Set the stat `stat` to `value`.
    pub fn set(&mut self, stat: Stat, value: Precision<ScalarValue>) {
        self.values.set(stat, value);
    }

    /// Clear the stat `stat` from the set.
    pub fn clear(&mut self, stat: Stat) {
        self.values.clear(stat);
    }
}

impl StatsProvider for MutTypedStatsSetRef<'_, '_> {
    fn get(&self, stat: Stat) -> Precision<Scalar> {
        self.values.get(stat).map(|sv| {
            Scalar::try_new(
                stat.dtype(self.dtype)
                    .vortex_expect("Must have valid dtype if value is present"),
                Some(sv),
            )
            .vortex_expect("failed to construct a scalar statistic")
        })
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

// Merge helpers
impl MutTypedStatsSetRef<'_, '_> {
    /// Merge stats set `other` into `self`, with the semantic assumption that `other`
    /// contains stats from a disjoint array that is *appended* to the array represented by `self`.
    pub fn merge_ordered(mut self, other: &TypedStatsSetRef) -> Self {
        for s in all::<Stat>() {
            match s {
                Stat::IsConstant => self.merge_is_constant(other),
                Stat::IsSorted => self.merge_is_sorted(other),
                Stat::IsStrictSorted => self.merge_is_strict_sorted(other),
                Stat::Max => self.merge_max(other),
                Stat::Min => self.merge_min(other),
                Stat::Sum => self.merge_sum(other),
                Stat::NullCount => self.merge_null_count(other),
                Stat::UncompressedSizeInBytes => self.merge_uncompressed_size_in_bytes(other),
                Stat::NaNCount => self.merge_nan_count(other),
            }
        }

        self
    }

    /// Merge stats set `other` into `self`, from a disjoint array, with no ordering assumptions.
    /// Stats that are not commutative (e.g., is_sorted) are dropped from the result.
    pub fn merge_unordered(mut self, other: &TypedStatsSetRef) -> Self {
        for s in all::<Stat>() {
            if !s.is_commutative() {
                self.clear(s);
                continue;
            }

            match s {
                Stat::IsConstant => self.merge_is_constant(other),
                Stat::Max => self.merge_max(other),
                Stat::Min => self.merge_min(other),
                Stat::Sum => self.merge_sum(other),
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

    /// Given two sets of stats (of differing precision) for the same array, combine them
    pub fn combine_sets(&mut self, other: &TypedStatsSetRef) -> VortexResult<()> {
        let other_stats: Vec<_> = other.values.iter().map(|(stat, _)| *stat).collect();
        for s in other_stats {
            match s {
                Stat::Max => self.combine_bound::<Max>(other)?,
                Stat::Min => self.combine_bound::<Min>(other)?,
                Stat::UncompressedSizeInBytes => {
                    self.combine_bound::<UncompressedSizeInBytes>(other)?
                }
                Stat::IsConstant => self.combine_bool_stat::<IsConstant>(other)?,
                Stat::IsSorted => self.combine_bool_stat::<IsSorted>(other)?,
                Stat::IsStrictSorted => self.combine_bool_stat::<IsStrictSorted>(other)?,
                Stat::NullCount => self.combine_bound::<NullCount>(other)?,
                Stat::Sum => self.combine_bound::<Sum>(other)?,
                Stat::NaNCount => self.combine_bound::<NaNCount>(other)?,
            }
        }
        Ok(())
    }

    fn combine_bound<S: StatType<Scalar>>(&mut self, other: &TypedStatsSetRef) -> VortexResult<()>
    where
        S::Bound: StatBound<Scalar> + Debug + Eq + PartialEq,
    {
        match (self.get_scalar_bound::<S>(), other.get_scalar_bound::<S>()) {
            (Some(m1), Some(m2)) => {
                let meet = m1
                    .intersection(&m2)
                    .vortex_expect("can always compare scalar")
                    .ok_or_else(|| {
                        vortex_err!("{:?} bounds ({m1:?}, {m2:?}) do not overlap", S::STAT)
                    })?;
                if meet != m1 {
                    self.set(
                        S::STAT,
                        meet.into_value().map(|s| {
                            s.into_value()
                                .vortex_expect("stat scalar value cannot be null")
                        }),
                    );
                }
            }
            (None, Some(m)) => self.set(
                S::STAT,
                m.into_value().map(|s| {
                    s.into_value()
                        .vortex_expect("stat scalar value cannot be null")
                }),
            ),
            (Some(_), _) => (),
            (None, None) => self.clear(S::STAT),
        }
        Ok(())
    }

    fn combine_bool_stat<S: StatType<bool>>(&mut self, other: &TypedStatsSetRef) -> VortexResult<()>
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

    fn merge_min(&mut self, other: &TypedStatsSetRef) {
        match (
            self.get_scalar_bound::<Min>(),
            other.get_scalar_bound::<Min>(),
        ) {
            (Some(m1), Some(m2)) => {
                let meet = m1.union(&m2).vortex_expect("can compare scalar");
                if meet != m1 {
                    self.set(
                        Stat::Min,
                        meet.into_value().map(|s| {
                            s.into_value()
                                .vortex_expect("stat scalar value cannot be null")
                        }),
                    );
                }
            }
            _ => self.clear(Stat::Min),
        }
    }

    fn merge_max(&mut self, other: &TypedStatsSetRef) {
        match (
            self.get_scalar_bound::<Max>(),
            other.get_scalar_bound::<Max>(),
        ) {
            (Some(m1), Some(m2)) => {
                let meet = m1.union(&m2).vortex_expect("can compare scalar");
                if meet != m1 {
                    self.set(
                        Stat::Max,
                        meet.into_value().map(|s| {
                            s.into_value()
                                .vortex_expect("stat scalar value cannot be null")
                        }),
                    );
                }
            }
            _ => self.clear(Stat::Max),
        }
    }

    fn merge_sum(&mut self, other: &TypedStatsSetRef) {
        match (
            self.get_scalar_bound::<Sum>(),
            other.get_scalar_bound::<Sum>(),
        ) {
            (Some(m1), Some(m2)) => {
                // If the combine sum is exact, then we can sum them.
                if let Some(scalar_value) =
                    m1.zip(m2).as_exact().and_then(|(s1, s2)| match s1.dtype() {
                        DType::Primitive(..) => s1
                            .as_primitive()
                            .checked_add(&s2.as_primitive())
                            .and_then(|pscalar| pscalar.pvalue().map(ScalarValue::Primitive)),
                        DType::Decimal(..) => s1
                            .as_decimal()
                            .checked_binary_numeric(
                                &s2.as_decimal(),
                                crate::scalar::NumericOperator::Add,
                            )
                            .map(|scalar| {
                                ScalarValue::Decimal(
                                    scalar
                                        .decimal_value()
                                        .vortex_expect("no decimal value in scalar"),
                                )
                            }),
                        _ => None,
                    })
                {
                    self.set(Stat::Sum, Precision::Exact(scalar_value));
                }
            }
            _ => self.clear(Stat::Sum),
        }
    }

    fn merge_is_constant(&mut self, other: &TypedStatsSetRef) {
        let self_const = self.get_as(Stat::IsConstant);
        let other_const = other.get_as(Stat::IsConstant);
        let self_min = self.get(Stat::Min);
        let other_min = other.get(Stat::Min);

        if let (Some(self_const), Some(other_const), Some(self_min), Some(other_min)) = (
            self_const.as_exact(),
            other_const.as_exact(),
            self_min.as_exact(),
            other_min.as_exact(),
        ) {
            if self_const && other_const && self_min == other_min {
                self.set(Stat::IsConstant, Precision::exact(true));
            } else {
                self.set(Stat::IsConstant, Precision::inexact(false));
            }
        }
        self.set(Stat::IsConstant, Precision::exact(false));
    }

    fn merge_is_sorted(&mut self, other: &TypedStatsSetRef) {
        self.merge_sortedness_stat(other, Stat::IsSorted, PartialOrd::le)
    }

    fn merge_is_strict_sorted(&mut self, other: &TypedStatsSetRef) {
        self.merge_sortedness_stat(other, Stat::IsStrictSorted, PartialOrd::lt)
    }

    fn merge_sortedness_stat<F: Fn(&Scalar, &Scalar) -> bool>(
        &mut self,
        other: &TypedStatsSetRef,
        stat: Stat,
        cmp: F,
    ) {
        if (Precision::Exact(true), Precision::Exact(true))
            == (self.get_as(stat), other.get_as(stat))
        {
            // There might be no stat because it was dropped, or it doesn't exist
            // (e.g. an all null array).
            // We assume that it was the dropped case since the doesn't exist might imply sorted,
            // but this in-precision is correct.
            if let (Some(self_max), Some(other_min)) = (
                self.get_scalar_bound::<Max>().and_then(|v| v.max_value()),
                other.get_scalar_bound::<Min>().and_then(|v| v.min_value()),
            ) {
                return if cmp(&self_max, &other_min) {
                    // keep value
                } else {
                    self.set(stat, Precision::inexact(false));
                };
            }
        }
        self.clear(stat);
    }

    fn merge_null_count(&mut self, other: &TypedStatsSetRef) {
        self.merge_sum_stat(Stat::NullCount, other)
    }

    fn merge_nan_count(&mut self, other: &TypedStatsSetRef) {
        self.merge_sum_stat(Stat::NaNCount, other)
    }

    fn merge_uncompressed_size_in_bytes(&mut self, other: &TypedStatsSetRef) {
        self.merge_sum_stat(Stat::UncompressedSizeInBytes, other)
    }

    fn merge_sum_stat(&mut self, stat: Stat, other: &TypedStatsSetRef) {
        let merged = self
            .get_as::<usize>(stat)
            .zip(other.get_as::<usize>(stat))
            .map(|(l, r)| ScalarValue::from(l + r));

        if merged.is_absent() {
            self.clear(stat);
        } else {
            self.set(stat, merged);
        }
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;
    use itertools::Itertools;
    use smallvec::smallvec;

    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::IsConstant;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::expr::stats::StatsProviderExt;
    use crate::stats::StatsSet;
    use crate::stats::stats_set::Scalar;

    #[test]
    fn test_iter() {
        // SAFETY: No duplicate stats.
        let set = unsafe {
            StatsSet::new_unchecked(smallvec![
                (Stat::Max, Precision::exact(100)),
                (Stat::Min, Precision::exact(42)),
            ])
        };
        let mut iter = set.iter();
        let first = iter.next().unwrap().clone();
        assert_eq!(first.0, Stat::Max);
        assert_eq!(
            first.1.map(
                |f| i32::try_from(&Scalar::try_new(PType::I32.into(), Some(f)).unwrap()).unwrap()
            ),
            Precision::exact(100)
        );
        let snd = iter.next().unwrap().clone();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(
            snd.1.map(
                |s| i32::try_from(&Scalar::try_new(PType::I32.into(), Some(s)).unwrap()).unwrap()
            ),
            Precision::exact(42)
        );
    }

    #[test]
    fn into_iter() {
        // SAFETY: No duplicate stats.
        let mut set = unsafe {
            StatsSet::new_unchecked(smallvec![
                (Stat::Max, Precision::exact(100)),
                (Stat::Min, Precision::exact(42)),
            ])
        }
        .into_iter();
        let (stat, first) = set.next().unwrap();
        assert_eq!(stat, Stat::Max);
        assert_eq!(
            first.map(
                |f| i32::try_from(&Scalar::try_new(PType::I32.into(), Some(f)).unwrap()).unwrap()
            ),
            Precision::exact(100)
        );
        let snd = set.next().unwrap();
        assert_eq!(snd.0, Stat::Min);
        assert_eq!(
            snd.1.map(
                |s| i32::try_from(&Scalar::try_new(PType::I32.into(), Some(s)).unwrap()).unwrap()
            ),
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

        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(
            first_ref.get_as::<bool>(Stat::IsConstant),
            Precision::exact(false)
        );
        assert_eq!(first_ref.get_as::<i32>(Stat::Min), Precision::exact(42));
    }

    #[test]
    fn merge_into_min() {
        let first = StatsSet::of(Stat::Min, Precision::exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );

        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(first_ref.get(Stat::Min).is_absent());
    }

    #[test]
    fn merge_from_min() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );

        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(first_ref.get(Stat::Min).is_absent());
    }

    #[test]
    fn merge_mins() {
        let first = StatsSet::of(Stat::Min, Precision::exact(37)).merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );

        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(first_ref.get_as::<i32>(Stat::Min), Precision::exact(37));
    }

    #[test]
    fn merge_into_bound_max() {
        let first = StatsSet::of(Stat::Max, Precision::exact(42)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_absent());
    }

    #[test]
    fn merge_from_max() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Max, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::Max).is_absent());
    }

    #[test]
    fn merge_maxes() {
        let first = StatsSet::of(Stat::Max, Precision::exact(37)).merge_ordered(
            &StatsSet::of(Stat::Max, Precision::exact(42)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(first_ref.get_as::<i32>(Stat::Max), Precision::exact(42));
    }

    #[test]
    fn merge_maxes_bound() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let first = StatsSet::of(Stat::Max, Precision::exact(42i32))
            .merge_ordered(&StatsSet::of(Stat::Max, Precision::inexact(43i32)), &dtype);
        let first_ref = first.as_typed_ref(&dtype);
        assert_eq!(first_ref.get_as::<i32>(Stat::Max), Precision::inexact(43));
    }

    #[test]
    fn merge_into_scalar() {
        // Sum stats for primitive types are always the 64-bit version (i64 for signed, u64
        // for unsigned, f64 for floats).
        let first = StatsSet::of(Stat::Sum, Precision::exact(42i64)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(first_ref.get(Stat::Sum).is_absent());
    }

    #[test]
    fn merge_from_scalar() {
        // Sum stats for primitive types are always the 64-bit version (i64 for signed, u64
        // for unsigned, f64 for floats).
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::Sum, Precision::exact(42i64)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert!(first_ref.get(Stat::Sum).is_absent());
    }

    #[test]
    fn merge_scalars() {
        // Sum stats for primitive types are always the 64-bit version (i64 for signed, u64
        // for unsigned, f64 for floats).
        let first = StatsSet::of(Stat::Sum, Precision::exact(37i64)).merge_ordered(
            &StatsSet::of(Stat::Sum, Precision::exact(42i64)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(first_ref.get_as::<i64>(Stat::Sum), Precision::exact(79i64));
    }

    #[test]
    fn merge_into_sortedness() {
        let first = StatsSet::of(Stat::IsStrictSorted, Precision::exact(true)).merge_ordered(
            &StatsSet::default(),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_absent());
    }

    #[test]
    fn merge_from_sortedness() {
        let first = StatsSet::default().merge_ordered(
            &StatsSet::of(Stat::IsStrictSorted, Precision::exact(true)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        assert!(first.get(Stat::IsStrictSorted).is_absent());
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

        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(
            first_ref.get_as::<bool>(Stat::IsStrictSorted),
            Precision::exact(true)
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

        let second_ref =
            second.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(
            second_ref.get_as::<bool>(Stat::IsStrictSorted),
            Precision::inexact(false)
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

        let second_ref =
            second.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(
            second_ref.get_as::<bool>(Stat::IsStrictSorted),
            Precision::exact(false)
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
        assert!(first.get(Stat::IsStrictSorted).is_absent());
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

        let first_ref = first.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(
            first_ref.get_as::<bool>(Stat::IsStrictSorted),
            Precision::exact(true)
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
        array
            .statistics()
            .compute_all(&all_stats, &mut array_session().create_execution_ctx())
            .unwrap();

        let stats = array.statistics().to_owned();
        for stat in &all_stats {
            assert!(!stats.get(*stat).is_absent(), "Stat {stat} is missing");
        }

        let merged = stats.clone().merge_unordered(
            &stats,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        for stat in &all_stats {
            assert_eq!(
                !merged.get(*stat).is_absent(),
                stat.is_commutative(),
                "Stat {stat} remains after merge_unordered despite not being commutative, or was removed despite being commutative"
            )
        }

        let merged_ref = merged.as_typed_ref(&DType::Primitive(PType::I32, Nullability::Nullable));
        let stats_ref = stats.as_typed_ref(&DType::Primitive(PType::I32, Nullability::Nullable));

        assert_eq!(
            merged_ref.get_as::<i32>(Stat::Min),
            stats_ref.get_as::<i32>(Stat::Min)
        );
        assert_eq!(
            merged_ref.get_as::<i32>(Stat::Max),
            stats_ref.get_as::<i32>(Stat::Max)
        );
        assert_eq!(
            merged_ref.get_as::<u64>(Stat::NullCount),
            stats_ref.get_as::<u64>(Stat::NullCount).map(|s| s * 2)
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
        let merged_ref =
            merged.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(merged_ref.get_as::<i32>(Stat::Min), Precision::exact(5));
    }

    #[test]
    fn merge_min_bound_bound_lower() {
        let merged = StatsSet::of(Stat::Min, Precision::inexact(4)).merge_ordered(
            &StatsSet::of(Stat::Min, Precision::exact(5)),
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        );
        let merged_ref =
            merged.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
        assert_eq!(merged_ref.get_as::<i32>(Stat::Min), Precision::inexact(4));
    }

    #[test]
    fn test_combine_is_constant() {
        {
            let mut stats = StatsSet::of(Stat::IsConstant, Precision::exact(true));
            let stats2 = StatsSet::of(Stat::IsConstant, Precision::exact(true));
            let mut stats_ref =
                stats.as_mut_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
            stats_ref
                .combine_bool_stat::<IsConstant>(
                    &stats2.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable)),
                )
                .unwrap();
            assert_eq!(
                stats_ref.get_as::<bool>(Stat::IsConstant),
                Precision::exact(true)
            );
        }

        {
            let mut stats = StatsSet::of(Stat::IsConstant, Precision::exact(true));
            let stats2 = StatsSet::of(Stat::IsConstant, Precision::inexact(false));
            let mut stats_ref =
                stats.as_mut_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
            stats_ref
                .combine_bool_stat::<IsConstant>(
                    &stats2.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable)),
                )
                .unwrap();
            assert_eq!(
                stats_ref.get_as::<bool>(Stat::IsConstant),
                Precision::exact(true)
            );
        }

        {
            let mut stats = StatsSet::of(Stat::IsConstant, Precision::exact(false));
            let stats2 = StatsSet::of(Stat::IsConstant, Precision::inexact(false));
            let mut stats_ref =
                stats.as_mut_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));
            stats_ref
                .combine_bool_stat::<IsConstant>(
                    &stats2.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable)),
                )
                .unwrap();
            assert_eq!(
                stats_ref.get_as::<bool>(Stat::IsConstant),
                Precision::exact(false)
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

        let stats_ref =
            stats1.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));

        // Min should remain unchanged
        assert_eq!(stats_ref.get_as::<i32>(Stat::Min), Precision::exact(42));
        // Max should be added
        assert_eq!(stats_ref.get_as::<i32>(Stat::Max), Precision::exact(100));
        // IsStrictSorted should be added
        assert_eq!(
            stats_ref.get_as::<bool>(Stat::IsStrictSorted),
            Precision::exact(true)
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

        let stats_ref =
            stats1.as_typed_ref(&DType::Primitive(PType::I32, Nullability::NonNullable));

        // Min should remain unchanged since it's more restrictive than the inexact value
        assert_eq!(stats_ref.get_as::<i32>(Stat::Min), Precision::exact(42));
        // Check that max was updated with the exact value
        assert_eq!(stats_ref.get_as::<i32>(Stat::Max), Precision::exact(90));
        // Check that IsSorted was added
        assert_eq!(
            stats_ref.get_as::<bool>(Stat::IsSorted),
            Precision::exact(true)
        );
    }
}
