use std::ops::BitAnd;

use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::{BoolArray, BoolEncoding};
use crate::stats::{Precision, Stat, StatsSet};
use crate::vtable::StatisticsVTable;

impl StatisticsVTable<&BoolArray> for BoolEncoding {
    fn compute_statistics(&self, array: &BoolArray, stat: Stat) -> VortexResult<StatsSet> {
        if array.is_empty() {
            return Ok(StatsSet::new_unchecked(vec![
                (Stat::Sum, Precision::exact(0)),
                (Stat::NullCount, Precision::exact(0)),
            ]));
        }

        match array.validity_mask()? {
            Mask::AllTrue(_) => self.compute_statistics(array.boolean_buffer(), stat),
            Mask::AllFalse(v) => Ok(StatsSet::nulls(v)),
            Mask::Values(values) => self.compute_statistics(
                &NullableBools(array.boolean_buffer(), values.boolean_buffer()),
                stat,
            ),
        }
    }
}

struct NullableBools<'a>(&'a BooleanBuffer, &'a BooleanBuffer);

impl StatisticsVTable<&NullableBools<'_>> for BoolEncoding {
    fn compute_statistics(&self, array: &NullableBools<'_>, stat: Stat) -> VortexResult<StatsSet> {
        // Fast-path if we just want the true-count
        if matches!(
            stat,
            Stat::Sum | Stat::Min | Stat::Max | Stat::IsConstant | Stat::NullCount
        ) {
            return Ok(StatsSet::bools_with_sum_and_null_count(
                array.0.bitand(array.1).count_set_bits(),
                array.1.len() - array.1.count_set_bits(),
                array.0.len(),
            ));
        }

        let first_non_null_idx = array
            .1
            .iter()
            .enumerate()
            .skip_while(|(_, valid)| !*valid)
            .map(|(idx, _)| idx)
            .next();

        if let Some(first_non_null) = first_non_null_idx {
            let mut acc = BoolStatsAccumulator::new(array.0.value(first_non_null));
            acc.n_nulls(first_non_null);
            array
                .0
                .iter()
                .zip_eq(array.1.iter())
                .skip(first_non_null + 1)
                .map(|(next, valid)| valid.then_some(next))
                .for_each(|next| acc.nullable_next(next));
            Ok(acc.finish())
        } else {
            Ok(StatsSet::nulls(array.0.len()))
        }
    }
}

impl StatisticsVTable<&BooleanBuffer> for BoolEncoding {
    fn compute_statistics(&self, buffer: &BooleanBuffer, stat: Stat) -> VortexResult<StatsSet> {
        // Fast-path if we just want the true-count
        if matches!(
            stat,
            Stat::Sum | Stat::Min | Stat::Max | Stat::IsConstant | Stat::NullCount
        ) {
            return Ok(StatsSet::bools_with_sum_and_null_count(
                buffer.count_set_bits(),
                0,
                buffer.len(),
            ));
        }

        let mut stats = BoolStatsAccumulator::new(buffer.value(0));
        buffer.iter().skip(1).for_each(|next| stats.next(next));
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
        StatsSet::new_unchecked(vec![
            (Stat::NullCount, Precision::exact(self.null_count)),
            (Stat::IsSorted, Precision::exact(self.is_sorted)),
            (
                Stat::IsStrictSorted,
                Precision::exact(
                    self.is_sorted && (self.len < 2 || (self.len == 2 && self.true_count == 1)),
                ),
            ),
        ])
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;

    use crate::ArrayVariants;
    use crate::array::Array;
    use crate::arrays::BoolArray;
    use crate::stats::Stat;
    use crate::validity::Validity;

    #[test]
    fn bool_stats() {
        let bool_arr = BoolArray::from_iter([false, false, true, true, false, true, true, false]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 0);
        assert_eq!(bool_arr.as_bool_typed().unwrap().true_count().unwrap(), 4);
    }

    #[test]
    fn strict_sorted() {
        let bool_arr_1 = BoolArray::from_iter([false, true]);
        assert!(bool_arr_1.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_1.statistics().compute_is_sorted().unwrap());

        let bool_arr_2 = BoolArray::from_iter([true]);
        assert!(bool_arr_2.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_2.statistics().compute_is_sorted().unwrap());

        let bool_arr_3 = BoolArray::from_iter([false]);
        assert!(bool_arr_3.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_3.statistics().compute_is_sorted().unwrap());

        let bool_arr_4 = BoolArray::from_iter([true, false]);
        assert!(!bool_arr_4.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr_4.statistics().compute_is_sorted().unwrap());

        let bool_arr_5 = BoolArray::from_iter([false, true, true]);
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
        assert_eq!(bool_arr.as_bool_typed().unwrap().true_count().unwrap(), 2);
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 3);
    }

    #[test]
    fn one_non_null_value() {
        let bool_arr = BoolArray::from_iter(vec![Some(false), None]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(!bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.as_bool_typed().unwrap().true_count().unwrap(), 0);
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 1);
    }

    #[test]
    fn empty_array() {
        let bool_arr = BoolArray::new(BooleanBuffer::new_set(0), Validity::NonNullable);
        assert!(bool_arr.statistics().compute_is_strict_sorted().is_none());
        assert!(bool_arr.statistics().compute_is_sorted().is_none());
        assert!(bool_arr.statistics().compute_is_constant().is_none());
        assert!(bool_arr.statistics().compute_min::<bool>().is_none());
        assert!(bool_arr.statistics().compute_max::<bool>().is_none());
        assert_eq!(bool_arr.as_bool_typed().unwrap().true_count().unwrap(), 0);
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 0);
    }

    #[test]
    fn all_false() {
        let bool_arr = BoolArray::from_iter(vec![false, false, false]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(!bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.as_bool_typed().unwrap().true_count().unwrap(), 0);
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 0);
    }

    #[test]
    fn all_nullable_stats() {
        let bool_arr = BoolArray::from_iter(vec![None, None, None, None, None]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_constant().unwrap());
        assert!(
            bool_arr
                .statistics()
                .compute_stat(Stat::Min)
                .unwrap()
                .is_none()
        );
        assert!(
            bool_arr
                .statistics()
                .compute_stat(Stat::Max)
                .unwrap()
                .is_none()
        );
        assert_eq!(bool_arr.as_bool_typed().unwrap().true_count().unwrap(), 0);
        assert_eq!(bool_arr.statistics().compute_null_count().unwrap(), 5);
    }
}
