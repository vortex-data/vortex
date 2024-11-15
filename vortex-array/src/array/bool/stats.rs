use std::ops::BitAnd;

use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::aliases::hash_map::HashMap;
use crate::array::BoolArray;
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::{ArrayDType, IntoArrayVariant};

impl ArrayStatisticsCompute for BoolArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::new());
        }

        match self.logical_validity() {
            LogicalValidity::AllValid(_) => self.boolean_buffer().compute_statistics(stat),
            LogicalValidity::AllInvalid(v) => Ok(StatsSet::nulls(v, self.dtype())),
            LogicalValidity::Array(a) => NullableBools(
                &self.boolean_buffer(),
                &a.clone().into_bool()?.boolean_buffer(),
            )
            .compute_statistics(stat),
        }
    }
}

struct NullableBools<'a>(&'a BooleanBuffer, &'a BooleanBuffer);

impl ArrayStatisticsCompute for NullableBools<'_> {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        // Fast-path if we just want the true-count
        if matches!(
            stat,
            Stat::TrueCount | Stat::Min | Stat::Max | Stat::IsConstant
        ) {
            return Ok(StatsSet::bools_with_true_count(
                self.0.bitand(self.1).count_set_bits(),
                self.0.len(),
            ));
        }

        let first_non_null_idx = self
            .1
            .iter()
            .enumerate()
            .skip_while(|(_, valid)| !*valid)
            .map(|(idx, _)| idx)
            .next();

        if let Some(first_non_null) = first_non_null_idx {
            let mut acc = BoolStatsAccumulator::new(self.0.value(first_non_null));
            acc.n_nulls(first_non_null);
            self.0
                .iter()
                .zip_eq(self.1.iter())
                .skip(first_non_null + 1)
                .map(|(next, valid)| valid.then_some(next))
                .for_each(|next| acc.nullable_next(next));
            Ok(acc.finish())
        } else {
            Ok(StatsSet::nulls(
                self.0.len(),
                &DType::Bool(Nullability::Nullable),
            ))
        }
    }
}

impl ArrayStatisticsCompute for BooleanBuffer {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        // Fast-path if we just want the true-count
        if matches!(
            stat,
            Stat::TrueCount | Stat::Min | Stat::Max | Stat::IsConstant
        ) {
            return Ok(StatsSet::bools_with_true_count(
                self.count_set_bits(),
                self.len(),
            ));
        }

        let mut stats = BoolStatsAccumulator::new(self.value(0));
        self.iter().skip(1).for_each(|next| stats.next(next));
        Ok(stats.finish())
    }
}

struct BoolStatsAccumulator {
    prev: bool,
    is_sorted: bool,
    run_count: usize,
    null_count: usize,
    true_count: usize,
    len: usize,
}

impl BoolStatsAccumulator {
    pub fn new(first_value: bool) -> Self {
        Self {
            prev: first_value,
            is_sorted: true,
            run_count: 1,
            null_count: 0,
            true_count: if first_value { 1 } else { 0 },
            len: 1,
        }
    }

    pub fn n_nulls(&mut self, n_nulls: usize) {
        self.null_count += n_nulls;
        self.len += n_nulls;
    }

    pub fn nullable_next(&mut self, next: Option<bool>) {
        match next {
            Some(n) => self.next(n),
            None => {
                self.null_count += 1;
                self.len += 1;
            }
        }
    }

    pub fn next(&mut self, next: bool) {
        self.len += 1;

        if next {
            self.true_count += 1
        }
        if !next & self.prev {
            self.is_sorted = false;
        }
        if next != self.prev {
            self.run_count += 1;
            self.prev = next;
        }
    }

    pub fn finish(self) -> StatsSet {
        StatsSet::from(HashMap::from([
            (Stat::NullCount, self.null_count.into()),
            (Stat::IsSorted, self.is_sorted.into()),
            (
                Stat::IsStrictSorted,
                (self.is_sorted && (self.len < 2 || (self.len == 2 && self.true_count == 1)))
                    .into(),
            ),
            (Stat::RunCount, self.run_count.into()),
        ]))
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::array::BoolArray;
    use crate::stats::{ArrayStatistics, Stat};

    #[test]
    fn bool_stats() {
        let bool_arr = BoolArray::from(vec![false, false, true, true, false, true, true, false]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 0);
        assert_eq!(bool_arr.statistics().compute_run_count().unwrap(), 5);
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 4);
    }

    #[test]
    fn strict_sorted() {
        let bool_arr_1 = BoolArray::from(vec![false, true]);
        assert!(bool_arr_1.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_1.statistics().compute_is_sorted().unwrap());

        let bool_arr_2 = BoolArray::from(vec![true]);
        assert!(bool_arr_2.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_2.statistics().compute_is_sorted().unwrap());

        let bool_arr_3 = BoolArray::from(vec![false]);
        assert!(bool_arr_3.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_3.statistics().compute_is_sorted().unwrap());

        let bool_arr_4 = BoolArray::from(vec![true, false]);
        assert!(!bool_arr_4.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr_4.statistics().compute_is_sorted().unwrap());

        let bool_arr_5 = BoolArray::from(vec![false, true, true]);
        assert!(!bool_arr_5.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_5.statistics().compute_is_sorted().unwrap());
    }

    #[test]
    fn nullable_stats() {
        let bool_arr = BoolArray::from_iter(vec![
            Some(false),
            Some(true),
            None,
            Some(true),
            Some(false),
            None,
            None,
        ]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.statistics().compute_run_count().unwrap(), 3);
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 2);
    }

    #[test]
    fn all_nullable_stats() {
        let bool_arr = BoolArray::from_iter(vec![None, None, None, None, None]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_constant().unwrap());
        assert_eq!(
            bool_arr.statistics().compute(Stat::Min).unwrap(),
            Scalar::null(DType::Bool(Nullable))
        );
        assert_eq!(
            bool_arr.statistics().compute(Stat::Max).unwrap(),
            Scalar::null(DType::Bool(Nullable))
        );
        assert_eq!(bool_arr.statistics().compute_run_count().unwrap(), 1);
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 0);
    }
}
