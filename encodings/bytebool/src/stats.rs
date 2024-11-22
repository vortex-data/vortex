use vortex_array::stats::{ArrayStatistics, Stat, StatisticsVTable, StatsSet};
use vortex_array::{ArrayLen, IntoArrayVariant};
use vortex_error::VortexResult;

use super::{ByteBoolArray, ByteBoolEncoding};

impl StatisticsVTable<ByteBoolArray> for ByteBoolEncoding {
    fn compute_statistics(&self, array: &ByteBoolArray, stat: Stat) -> VortexResult<StatsSet> {
        if array.is_empty() {
            return Ok(StatsSet::default());
        }

        // TODO(adamgs): This is slightly wasteful and could be optimized in the future
        let bools = array.as_ref().clone().into_bool()?;
        Ok(StatsSet::from_iter(
            bools
                .statistics()
                .compute(stat)
                .into_iter()
                .map(|value| (stat, value)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::stats::ArrayStatistics;
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn bool_stats() {
        let bool_arr =
            ByteBoolArray::from(vec![false, false, true, true, false, true, true, false]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.statistics().compute_run_count().unwrap(), 5);
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 4);
    }

    #[test]
    fn strict_sorted() {
        let bool_arr_1 = ByteBoolArray::from(vec![false, true]);
        assert!(bool_arr_1.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_1.statistics().compute_is_sorted().unwrap());

        let bool_arr_2 = ByteBoolArray::from(vec![true]);
        assert!(bool_arr_2.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_2.statistics().compute_is_sorted().unwrap());

        let bool_arr_3 = ByteBoolArray::from(vec![false]);
        assert!(bool_arr_3.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_3.statistics().compute_is_sorted().unwrap());

        let bool_arr_4 = ByteBoolArray::from(vec![true, false]);
        assert!(!bool_arr_4.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr_4.statistics().compute_is_sorted().unwrap());

        let bool_arr_5 = ByteBoolArray::from(vec![false, true, true]);
        assert!(!bool_arr_5.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_5.statistics().compute_is_sorted().unwrap());
    }

    #[test]
    fn nullable_stats() {
        let bool_arr = ByteBoolArray::from(vec![
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
        let bool_arr = ByteBoolArray::from(vec![None, None, None, None, None]);
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(bool_arr.statistics().compute_is_constant().unwrap());
        assert_eq!(
            bool_arr.statistics().compute(Stat::Min).unwrap(),
            Scalar::null(DType::Bool(Nullability::Nullable))
        );
        assert_eq!(
            bool_arr.statistics().compute(Stat::Max).unwrap(),
            Scalar::null(DType::Bool(Nullability::Nullable))
        );
        assert_eq!(bool_arr.statistics().compute_run_count().unwrap(), 1);
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 0);
    }
}
