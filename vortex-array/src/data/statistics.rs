use enum_iterator::all;
use itertools::Itertools;
use vortex_error::{vortex_panic, VortexExpect as _};
use vortex_scalar::ScalarValue;

use crate::data::InnerArrayData;
use crate::stats::{Stat, Statistics, StatsSet};
use crate::ArrayData;

impl Statistics for ArrayData {
    fn get(&self, stat: Stat) -> Option<ScalarValue> {
        match &self.0 {
            InnerArrayData::Owned(o) => o
                .stats_set
                .read()
                .unwrap_or_else(|_| {
                    vortex_panic!(
                        "Failed to acquire read lock on stats map while getting {}",
                        stat
                    )
                })
                .get(stat)
                .cloned(),
            InnerArrayData::Viewed(v) => match stat {
                Stat::Max => {
                    let max = v.flatbuffer().stats()?.max();
                    max.and_then(|v| ScalarValue::try_from(v).ok())
                }
                Stat::Min => {
                    let min = v.flatbuffer().stats()?.min();
                    min.and_then(|v| ScalarValue::try_from(v).ok())
                }
                Stat::IsConstant => v.flatbuffer().stats()?.is_constant().map(bool::into),
                Stat::IsSorted => v.flatbuffer().stats()?.is_sorted().map(bool::into),
                Stat::IsStrictSorted => v.flatbuffer().stats()?.is_strict_sorted().map(bool::into),
                Stat::RunCount => v.flatbuffer().stats()?.run_count().map(u64::into),
                Stat::TrueCount => v.flatbuffer().stats()?.true_count().map(u64::into),
                Stat::NullCount => v.flatbuffer().stats()?.null_count().map(u64::into),
                Stat::BitWidthFreq => v
                    .flatbuffer()
                    .stats()?
                    .bit_width_freq()
                    .map(|v| v.iter().collect_vec())
                    .map(ScalarValue::from),
                Stat::TrailingZeroFreq => v
                    .flatbuffer()
                    .stats()?
                    .trailing_zero_freq()
                    .map(|v| v.iter().collect_vec())
                    .map(ScalarValue::from),
                Stat::UncompressedSizeInBytes => v
                    .flatbuffer()
                    .stats()?
                    .uncompressed_size_in_bytes()
                    .map(u64::into),
            },
        }
    }

    fn to_set(&self) -> StatsSet {
        match &self.0 {
            InnerArrayData::Owned(o) => o
                .stats_set
                .read()
                .unwrap_or_else(|_| vortex_panic!("Failed to acquire read lock on stats map"))
                .clone(),
            InnerArrayData::Viewed(_) => StatsSet::from_iter(
                all::<Stat>().filter_map(|stat| self.get(stat).map(|v| (stat, v))),
            ),
        }
    }

    fn set(&self, stat: Stat, value: ScalarValue) {
        match &self.0 {
            InnerArrayData::Owned(o) => o
                .stats_set
                .write()
                .unwrap_or_else(|_| {
                    vortex_panic!(
                        "Failed to acquire write lock on stats map while setting {} to {}",
                        stat,
                        value
                    )
                })
                .set(stat, value),
            InnerArrayData::Viewed(_) => {
                // We cannot modify stats on a view
            }
        }
    }

    fn clear(&self, stat: Stat) {
        match &self.0 {
            InnerArrayData::Owned(o) => {
                o.stats_set
                    .write()
                    .unwrap_or_else(|_| vortex_panic!("Failed to acquire write lock on stats map"))
                    .clear(stat);
            }
            InnerArrayData::Viewed(_) => {
                // We cannot modify stats on a view
            }
        }
    }

    fn compute(&self, stat: Stat) -> Option<ScalarValue> {
        if let Some(s) = self.get(stat) {
            return Some(s);
        }
        let s = self
            .encoding()
            .compute_statistics(self, stat)
            .vortex_expect("compute_statistics must not fail")
            .get(stat)
            .cloned();

        if let Some(s) = &s {
            self.set(stat, s.clone());
        }

        s
    }

    fn retain_only(&self, stats: &[Stat]) {
        match &self.0 {
            InnerArrayData::Owned(o) => {
                o.stats_set
                    .write()
                    .unwrap_or_else(|_| vortex_panic!("Failed to acquire write lock on stats map"))
                    .retain_only(stats);
            }
            InnerArrayData::Viewed(_) => {
                // We cannot modify stats on a view
            }
        }
    }
}
