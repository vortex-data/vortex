use vortex_array::stats::{Stat, StatisticsVTable, StatsSet};
use vortex_array::ArrayLen;
use vortex_error::{vortex_err, VortexResult};

use crate::{RoaringBoolArray, RoaringBoolEncoding};

impl StatisticsVTable<RoaringBoolArray> for RoaringBoolEncoding {
    fn compute_statistics(&self, array: &RoaringBoolArray, stat: Stat) -> VortexResult<StatsSet> {
        // Only needs to compute IsSorted, IsStrictSorted and RunCount all other stats have been populated on construction
        let bitmap = array.bitmap();
        let true_count = bitmap.statistics().cardinality;
        if matches!(
            stat,
            Stat::TrueCount | Stat::Min | Stat::Max | Stat::IsConstant
        ) {
            return Ok(StatsSet::bools_with_true_and_null_count(
                true_count as usize,
                0,
                array.len(),
            ));
        }

        if matches!(stat, Stat::IsSorted | Stat::IsStrictSorted) {
            let is_sorted = if true_count == 0 || true_count == array.len() as u64 {
                true
            } else {
                let min_idx = bitmap.minimum().ok_or_else(|| {
                    vortex_err!("Bitmap has no minimum despite having cardinality > 0")
                })?;
                let max_idx = bitmap.maximum().ok_or_else(|| {
                    vortex_err!("Bitmap has no maximum despite having cardinality > 0")
                })?;
                (max_idx as usize + 1 == array.len())
                    && (max_idx + 1 - min_idx) as u64 == true_count
            };

            let is_strict_sorted =
                is_sorted && (array.len() <= 1 || (array.len() == 2 && true_count == 1));
            return Ok(StatsSet::from_iter([
                (Stat::IsSorted, is_sorted.into()),
                (Stat::IsStrictSorted, is_strict_sorted.into()),
            ]));
        }

        Ok(StatsSet::default())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::BoolArray;
    use vortex_array::stats::ArrayStatistics;
    use vortex_array::IntoArrayData;

    use crate::RoaringBoolArray;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bool_stats() {
        let bool_arr = RoaringBoolArray::encode(
            BoolArray::from_iter([false, false, true, true, false, true, true, false]).into_array(),
        )
        .unwrap();
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 4);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn strict_sorted() {
        let bool_arr_1 =
            RoaringBoolArray::encode(BoolArray::from_iter([false, true]).into_array()).unwrap();
        assert!(bool_arr_1.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_1.statistics().compute_is_sorted().unwrap());

        let bool_arr_2 =
            RoaringBoolArray::encode(BoolArray::from_iter([true]).into_array()).unwrap();
        assert!(bool_arr_2.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_2.statistics().compute_is_sorted().unwrap());

        let bool_arr_3 =
            RoaringBoolArray::encode(BoolArray::from_iter([false]).into_array()).unwrap();
        assert!(bool_arr_3.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_3.statistics().compute_is_sorted().unwrap());

        let bool_arr_4 =
            RoaringBoolArray::encode(BoolArray::from_iter([true, false]).into_array()).unwrap();
        assert!(!bool_arr_4.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr_4.statistics().compute_is_sorted().unwrap());

        let bool_arr_5 =
            RoaringBoolArray::encode(BoolArray::from_iter([false, true, true]).into_array())
                .unwrap();
        assert!(!bool_arr_5.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_5.statistics().compute_is_sorted().unwrap());
    }
}
