use croaring::Bitset;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use vortex_error::{vortex_err, VortexResult};

use crate::RoaringBoolArray;

impl ArrayStatisticsCompute for RoaringBoolArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::new());
        }

        // Only needs to compute IsSorted, IsStrictSorted and RunCount all other stats have been populated on construction
        let bitmap = self.bitmap();
        BitmapStats(
            bitmap
                .to_bitset()
                .ok_or_else(|| vortex_err!("Bitmap to Bitset conversion run out of memory"))?,
            self.len(),
            bitmap.statistics().cardinality,
        )
        .compute_statistics(stat)
    }
}

// Underlying bitset, length in bits, cardinality (true count) of the bitset
struct BitmapStats(Bitset, usize, u64);

impl ArrayStatisticsCompute for BitmapStats {
    fn compute_statistics(&self, _stat: Stat) -> VortexResult<StatsSet> {
        let bitset_slice = self.0.as_slice();
        let whole_chunks = self.1 / 64;
        let last_chunk_len = self.1 % 64;
        let fist_bool = bitset_slice[0] & 1 == 1;
        let mut stats = RoaringBoolStatsAccumulator::new(fist_bool);
        for bits64 in bitset_slice[0..whole_chunks].iter() {
            stats.next(*bits64);
        }
        stats.next_up_to_length(bitset_slice[whole_chunks], last_chunk_len);
        Ok(stats.finish(self.2))
    }
}

struct RoaringBoolStatsAccumulator {
    prev: bool,
    is_sorted: bool,
    run_count: usize,
    len: usize,
}

impl RoaringBoolStatsAccumulator {
    fn new(first_value: bool) -> Self {
        Self {
            prev: first_value,
            is_sorted: true,
            run_count: 1,
            len: 0,
        }
    }

    pub fn next_up_to_length(&mut self, next: u64, len: usize) {
        assert!(len <= 64);
        self.len += len;
        for i in 0..len {
            let current = ((next >> i) & 1) == 1;
            // Booleans are sorted true > false so we aren't sorted if we switched from true to false value
            if !current && self.prev {
                self.is_sorted = false;
            }
            if current != self.prev {
                self.run_count += 1;
                self.prev = current;
            }
        }
    }

    pub fn next(&mut self, next: u64) {
        self.next_up_to_length(next, 64)
    }

    pub fn finish(self, cardinality: u64) -> StatsSet {
        StatsSet::from(HashMap::from([
            (Stat::IsSorted, self.is_sorted.into()),
            (
                Stat::IsStrictSorted,
                (self.is_sorted && (self.len < 2 || (self.len == 2 && cardinality == 1))).into(),
            ),
            (Stat::RunCount, self.run_count.into()),
        ]))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::array::BoolArray;
    use vortex_array::stats::ArrayStatistics;
    use vortex_array::IntoArray;

    use crate::RoaringBoolArray;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn bool_stats() {
        let bool_arr = RoaringBoolArray::encode(
            BoolArray::from(vec![false, false, true, true, false, true, true, false]).into_array(),
        )
        .unwrap();
        assert!(!bool_arr.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_sorted().unwrap());
        assert!(!bool_arr.statistics().compute_is_constant().unwrap());
        assert!(!bool_arr.statistics().compute_min::<bool>().unwrap());
        assert!(bool_arr.statistics().compute_max::<bool>().unwrap());
        assert_eq!(bool_arr.statistics().compute_run_count().unwrap(), 5);
        assert_eq!(bool_arr.statistics().compute_true_count().unwrap(), 4);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn strict_sorted() {
        let bool_arr_1 =
            RoaringBoolArray::encode(BoolArray::from(vec![false, true]).into_array()).unwrap();
        assert!(bool_arr_1.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_1.statistics().compute_is_sorted().unwrap());

        let bool_arr_2 =
            RoaringBoolArray::encode(BoolArray::from(vec![true]).into_array()).unwrap();
        assert!(bool_arr_2.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_2.statistics().compute_is_sorted().unwrap());

        let bool_arr_3 =
            RoaringBoolArray::encode(BoolArray::from(vec![false]).into_array()).unwrap();
        assert!(bool_arr_3.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_3.statistics().compute_is_sorted().unwrap());

        let bool_arr_4 =
            RoaringBoolArray::encode(BoolArray::from(vec![true, false]).into_array()).unwrap();
        assert!(!bool_arr_4.statistics().compute_is_strict_sorted().unwrap());
        assert!(!bool_arr_4.statistics().compute_is_sorted().unwrap());

        let bool_arr_5 =
            RoaringBoolArray::encode(BoolArray::from(vec![false, true, true]).into_array())
                .unwrap();
        assert!(!bool_arr_5.statistics().compute_is_strict_sorted().unwrap());
        assert!(bool_arr_5.statistics().compute_is_sorted().unwrap());
    }
}
