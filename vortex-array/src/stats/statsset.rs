use enum_iterator::all;
use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexError, VortexExpect};
use vortex_scalar::{Scalar, ScalarValue};

use crate::aliases::hash_map::{Entry, HashMap, IntoIter};
use crate::stats::Stat;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatsSet {
    values: HashMap<Stat, Scalar>,
}

impl From<HashMap<Stat, Scalar>> for StatsSet {
    fn from(value: HashMap<Stat, Scalar>) -> Self {
        Self { values: value }
    }
}

impl StatsSet {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Specialized constructor for the case where the StatsSet represents
    /// an array consisting entirely of [null](vortex_dtype::DType::Null) values.
    pub fn nulls(len: usize, dtype: &DType) -> Self {
        let mut stats = HashMap::from([
            (Stat::Min, Scalar::null(dtype.clone())),
            (Stat::Max, Scalar::null(dtype.clone())),
            (Stat::IsConstant, true.into()),
            (Stat::IsSorted, true.into()),
            (Stat::IsStrictSorted, (len < 2).into()),
            (Stat::RunCount, 1.into()),
            (Stat::NullCount, len.into()),
        ]);

        // Add any DType-specific stats.
        match dtype {
            DType::Bool(_) => {
                stats.insert(Stat::TrueCount, 0.into());
            }
            DType::Primitive(ptype, _) => {
                ptype.byte_width();
                stats.insert(
                    Stat::BitWidthFreq,
                    vec![0_u64; ptype.byte_width() * 8 + 1].into(),
                );
                stats.insert(
                    Stat::TrailingZeroFreq,
                    vec![ptype.byte_width() * 8; ptype.byte_width() * 8 + 1].into(),
                );
            }
            _ => {}
        }

        Self::from(stats)
    }

    pub fn constant(scalar: Scalar, length: usize) -> Self {
        let mut stats = Self::new();
        stats.set(Stat::IsConstant, true.into());
        stats.set(Stat::IsSorted, true.into());
        stats.set(Stat::IsStrictSorted, (length <= 1).into());

        let run_count = if length == 0 { 0 } else { 1 };
        stats.set(Stat::RunCount, run_count.into());

        let null_count = if scalar.value().is_null() {
            length as u64
        } else {
            0
        };
        stats.set(Stat::NullCount, null_count.into());

        if let ScalarValue::Bool(b) = scalar.value() {
            let true_count = if *b { length as u64 } else { 0 };
            stats.set(Stat::TrueCount, true_count.into());
        }

        stats.set(Stat::Min, scalar.clone());
        stats.set(Stat::Max, scalar);

        stats
    }

    pub fn bools_with_true_count(true_count: usize, len: usize) -> StatsSet {
        StatsSet::from(HashMap::<Stat, Scalar>::from([
            (Stat::TrueCount, true_count.into()),
            (Stat::Min, (true_count == len).into()),
            (Stat::Max, (true_count > 0).into()),
            (
                Stat::IsConstant,
                (true_count == 0 || true_count == len).into(),
            ),
        ]))
    }

    pub fn of<S: Into<Scalar>>(stat: Stat, value: S) -> Self {
        Self::from(HashMap::from([(stat, value.into())]))
    }

    pub fn get(&self, stat: Stat) -> Option<&Scalar> {
        self.values.get(&stat)
    }

    fn get_as<T: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(&self, stat: Stat) -> Option<T> {
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

    pub fn set(&mut self, stat: Stat, value: Scalar) {
        self.values.insert(stat, value);
    }

    /// Merge stats set `other` into `self`, with the semantic assumption that `other`
    /// contains stats from an array that is *appended* to the array represented by `self`.
    pub fn merge_ordered(&mut self, other: &Self) -> &Self {
        for s in all::<Stat>() {
            match s {
                Stat::BitWidthFreq => self.merge_bit_width_freq(other),
                Stat::TrailingZeroFreq => self.merge_trailing_zero_freq(other),
                Stat::IsConstant => self.merge_is_constant(other),
                Stat::IsSorted => self.merge_is_sorted(other),
                Stat::IsStrictSorted => self.merge_is_strict_sorted(other),
                Stat::Max => self.merge_max(other),
                Stat::Min => self.merge_min(other),
                Stat::RunCount => self.merge_run_count(other),
                Stat::TrueCount => self.merge_true_count(other),
                Stat::NullCount => self.merge_null_count(other),
            }
        }

        self
    }

    /// Merge stats set `other` into `self`, with no assumption on ordering.
    /// Stats that are not commutative (e.g., is_sorted) are dropped from the result.
    pub fn merge_unordered(&mut self, other: &Self) -> &Self {
        for s in all::<Stat>() {
            if !s.is_commutative() {
                self.values.remove(&s);
                continue;
            }

            match s {
                Stat::BitWidthFreq => self.merge_bit_width_freq(other),
                Stat::TrailingZeroFreq => self.merge_trailing_zero_freq(other),
                Stat::IsConstant => self.merge_is_constant(other),
                Stat::Max => self.merge_max(other),
                Stat::Min => self.merge_min(other),
                Stat::TrueCount => self.merge_true_count(other),
                Stat::NullCount => self.merge_null_count(other),
                _ => vortex_panic!("Unrecognized commutative stat {}", s),
            }
        }

        self
    }

    fn merge_min(&mut self, other: &Self) {
        self.merge_scalars(Stat::Min, other, |other, own| other < own);
    }

    fn merge_max(&mut self, other: &Self) {
        self.merge_scalars(Stat::Max, other, |other, own| other > own);
    }

    /// Merges stats if both are present, if either stat is not present, drops the stat from the
    /// result set. For example, if we know the minimums of two arrays, the minimum of their union
    /// is the minimum-of-minimums, but if we only know the minimum of one of the two arrays, we
    /// do not know the minimum of their union.
    fn merge_scalars<F: Fn(&Scalar, &Scalar) -> bool>(&mut self, stat: Stat, other: &Self, cmp: F) {
        if let Entry::Occupied(mut e) = self.values.entry(stat) {
            if let Some(ov) = other.get(stat) {
                if cmp(ov, e.get()) {
                    e.insert(ov.clone());
                }
            } else {
                e.remove();
            }
        }
    }

    fn merge_is_constant(&mut self, other: &Self) {
        if let Some(is_constant) = self.get_as(Stat::IsConstant) {
            if let Some(other_is_constant) = other.get_as(Stat::IsConstant) {
                if is_constant && other_is_constant && self.get(Stat::Min) == other.get(Stat::Min) {
                    return;
                }
            }
            self.values.insert(Stat::IsConstant, false.into());
        }
    }

    fn merge_is_sorted(&mut self, other: &Self) {
        self.merge_sortedness_stat(other, Stat::IsSorted, |own, other| own <= other)
    }

    fn merge_is_strict_sorted(&mut self, other: &Self) {
        self.merge_sortedness_stat(other, Stat::IsStrictSorted, |own, other| own < other)
    }

    fn merge_sortedness_stat<F: Fn(Option<&Scalar>, Option<&Scalar>) -> bool>(
        &mut self,
        other: &Self,
        stat: Stat,
        cmp: F,
    ) {
        if let Some(is_sorted) = self.get_as(stat) {
            if let Some(other_is_sorted) = other.get_as(stat) {
                if !(self.get(Stat::Max).is_some() && other.get(Stat::Min).is_some()) {
                    self.values.remove(&stat);
                } else if is_sorted
                    && other_is_sorted
                    && cmp(self.get(Stat::Max), other.get(Stat::Min))
                {
                    return;
                } else {
                    self.values.insert(stat, false.into());
                }
            } else {
                self.values.remove(&stat);
            }
        }
    }

    fn merge_true_count(&mut self, other: &Self) {
        self.merge_scalar_stat(other, Stat::TrueCount)
    }

    fn merge_null_count(&mut self, other: &Self) {
        self.merge_scalar_stat(other, Stat::NullCount)
    }

    fn merge_scalar_stat(&mut self, other: &Self, stat: Stat) {
        if let Entry::Occupied(mut e) = self.values.entry(stat) {
            if let Some(other_value) = other.get_as::<usize>(stat) {
                let self_value: usize = e.get().try_into().unwrap_or_else(|err: VortexError| {
                    vortex_panic!(err, "Failed to get stat {} as usize", stat)
                });
                e.insert((self_value + other_value).into());
            } else {
                e.remove();
            }
        }
    }

    fn merge_bit_width_freq(&mut self, other: &Self) {
        self.merge_freq_stat(other, Stat::BitWidthFreq)
    }

    fn merge_trailing_zero_freq(&mut self, other: &Self) {
        self.merge_freq_stat(other, Stat::TrailingZeroFreq)
    }

    fn merge_freq_stat(&mut self, other: &Self, stat: Stat) {
        if let Entry::Occupied(mut e) = self.values.entry(stat) {
            if let Some(other_value) = other.get_as::<Vec<u64>>(stat) {
                // TODO(robert): Avoid the copy here. We could e.get_mut() but need to figure out casting
                let self_value: Vec<u64> = e.get().try_into().unwrap_or_else(|err: VortexError| {
                    vortex_panic!(err, "Failed to get stat {} as Vec<u64>", stat)
                });
                e.insert(
                    self_value
                        .iter()
                        .zip_eq(other_value.iter())
                        .map(|(s, o)| *s + *o)
                        .collect::<Vec<_>>()
                        .into(),
                );
            } else {
                e.remove();
            }
        }
    }

    /// Merged run count is an upper bound where we assume run is interrupted at the boundary
    fn merge_run_count(&mut self, other: &Self) {
        if let Entry::Occupied(mut e) = self.values.entry(Stat::RunCount) {
            if let Some(other_value) = other.get_as::<usize>(Stat::RunCount) {
                let self_value: usize = e
                    .get()
                    .try_into()
                    .vortex_expect("Failed to get run count as usize");
                e.insert((self_value + other_value + 1).into());
            } else {
                e.remove();
            }
        }
    }
}

impl Extend<(Stat, Scalar)> for StatsSet {
    #[inline]
    fn extend<T: IntoIterator<Item = (Stat, Scalar)>>(&mut self, iter: T) {
        self.values.extend(iter)
    }
}

impl IntoIterator for StatsSet {
    type Item = (Stat, Scalar);
    type IntoIter = IntoIter<Stat, Scalar>;

    fn into_iter(self) -> IntoIter<Stat, Scalar> {
        self.values.into_iter()
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;
    use itertools::Itertools;

    use crate::array::PrimitiveArray;
    use crate::stats::{ArrayStatistics as _, Stat, StatsSet};
    use crate::IntoArrayData as _;

    #[test]
    fn merge_into_min() {
        let mut first = StatsSet::of(Stat::Min, 42);
        first.merge_ordered(&StatsSet::new());
        assert_eq!(first.get(Stat::Min), None);
    }

    #[test]
    fn merge_from_min() {
        let mut first = StatsSet::new();
        first.merge_ordered(&StatsSet::of(Stat::Min, 42));
        assert_eq!(first.get(Stat::Min), None);
    }

    #[test]
    fn merge_mins() {
        let mut first = StatsSet::of(Stat::Min, 37);
        first.merge_ordered(&StatsSet::of(Stat::Min, 42));
        assert_eq!(first.get(Stat::Min).cloned(), Some(37.into()));
    }

    #[test]
    fn merge_into_max() {
        let mut first = StatsSet::of(Stat::Max, 42);
        first.merge_ordered(&StatsSet::new());
        assert_eq!(first.get(Stat::Max), None);
    }

    #[test]
    fn merge_from_max() {
        let mut first = StatsSet::new();
        first.merge_ordered(&StatsSet::of(Stat::Max, 42));
        assert_eq!(first.get(Stat::Max), None);
    }

    #[test]
    fn merge_maxes() {
        let mut first = StatsSet::of(Stat::Max, 37);
        first.merge_ordered(&StatsSet::of(Stat::Max, 42));
        assert_eq!(first.get(Stat::Max).cloned(), Some(42.into()));
    }

    #[test]
    fn merge_into_scalar() {
        let mut first = StatsSet::of(Stat::TrueCount, 42);
        first.merge_ordered(&StatsSet::new());
        assert_eq!(first.get(Stat::TrueCount), None);
    }

    #[test]
    fn merge_from_scalar() {
        let mut first = StatsSet::new();
        first.merge_ordered(&StatsSet::of(Stat::TrueCount, 42));
        assert_eq!(first.get(Stat::TrueCount), None);
    }

    #[test]
    fn merge_scalars() {
        let mut first = StatsSet::of(Stat::TrueCount, 37);
        first.merge_ordered(&StatsSet::of(Stat::TrueCount, 42));
        assert_eq!(first.get(Stat::TrueCount).cloned(), Some(79u64.into()));
    }

    #[test]
    fn merge_into_freq() {
        let vec = (0..255).collect_vec();
        let mut first = StatsSet::of(Stat::BitWidthFreq, vec);
        first.merge_ordered(&StatsSet::new());
        assert_eq!(first.get(Stat::BitWidthFreq), None);
    }

    #[test]
    fn merge_from_freq() {
        let vec = (0..255).collect_vec();
        let mut first = StatsSet::new();
        first.merge_ordered(&StatsSet::of(Stat::BitWidthFreq, vec));
        assert_eq!(first.get(Stat::BitWidthFreq), None);
    }

    #[test]
    fn merge_freqs() {
        let vec_in = vec![5u64; 256];
        let vec_out = vec![10u64; 256];
        let mut first = StatsSet::of(Stat::BitWidthFreq, vec_in.clone());
        first.merge_ordered(&StatsSet::of(Stat::BitWidthFreq, vec_in));
        assert_eq!(first.get(Stat::BitWidthFreq).cloned(), Some(vec_out.into()));
    }

    #[test]
    fn merge_into_sortedness() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.merge_ordered(&StatsSet::new());
        assert_eq!(first.get(Stat::IsStrictSorted), None);
    }

    #[test]
    fn merge_from_sortedness() {
        let mut first = StatsSet::new();
        first.merge_ordered(&StatsSet::of(Stat::IsStrictSorted, true));
        assert_eq!(first.get(Stat::IsStrictSorted), None);
    }

    #[test]
    fn merge_sortedness() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1.into());
        let mut second = StatsSet::of(Stat::IsStrictSorted, true);
        second.set(Stat::Min, 2.into());
        first.merge_ordered(&second);
        assert_eq!(first.get(Stat::IsStrictSorted).cloned(), Some(true.into()));
    }

    #[test]
    fn merge_sortedness_out_of_order() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Min, 1.into());
        let mut second = StatsSet::of(Stat::IsStrictSorted, true);
        second.set(Stat::Max, 2.into());
        second.merge_ordered(&first);
        assert_eq!(
            second.get(Stat::IsStrictSorted).cloned(),
            Some(false.into())
        );
    }

    #[test]
    fn merge_sortedness_only_one_sorted() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1.into());
        let mut second = StatsSet::of(Stat::IsStrictSorted, false);
        second.set(Stat::Min, 2.into());
        first.merge_ordered(&second);
        assert_eq!(
            second.get(Stat::IsStrictSorted).cloned(),
            Some(false.into())
        );
    }

    #[test]
    fn merge_sortedness_missing_min() {
        let mut first = StatsSet::of(Stat::IsStrictSorted, true);
        first.set(Stat::Max, 1.into());
        let second = StatsSet::of(Stat::IsStrictSorted, true);
        first.merge_ordered(&second);
        assert_eq!(first.get(Stat::IsStrictSorted).cloned(), None);
    }

    #[test]
    fn merge_unordered() {
        let array = PrimitiveArray::from_nullable_vec(vec![
            Some(1),
            None,
            Some(2),
            Some(42),
            Some(10000),
            None,
        ])
        .into_array();
        let all_stats = all::<Stat>()
            .filter(|s| !matches!(s, Stat::TrueCount))
            .collect_vec();
        array.statistics().compute_all(&all_stats).unwrap();

        let stats = array.statistics().to_set();
        for stat in &all_stats {
            assert!(stats.get(*stat).is_some(), "Stat {} is missing", stat);
        }

        let mut merged = stats.clone();
        merged.merge_unordered(&stats);
        for stat in &all_stats {
            assert_eq!(
                merged.get(*stat).is_some(),
                stat.is_commutative(),
                "Stat {} remains after merge_unordered despite not being commutative, or was removed despite being commutative",
                stat
            )
        }

        assert_eq!(merged.get(Stat::Min), stats.get(Stat::Min));
        assert_eq!(merged.get(Stat::Max), stats.get(Stat::Max));
        assert_eq!(
            merged
                .get(Stat::NullCount)
                .unwrap()
                .value()
                .as_pvalue()
                .unwrap()
                .unwrap()
                .as_u64()
                .unwrap(),
            2 * stats
                .get(Stat::NullCount)
                .unwrap()
                .value()
                .as_pvalue()
                .unwrap()
                .unwrap()
                .as_u64()
                .unwrap()
        );
    }
}
