use enum_iterator::all;
use itertools::Itertools;
use vortex_error::{vortex_panic, VortexExpect as _};
use vortex_scalar::ScalarValue;

use crate::data::InnerArray;
use crate::stats::{Precision, Stat, Statistics, StatsSet};
use crate::Array;

impl Statistics for Array {
    fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        match &self.0 {
            InnerArray::Owned(o) => o
                .stats_set
                .read()
                .unwrap_or_else(|_| {
                    vortex_panic!(
                        "Failed to acquire read lock on stats map while getting {}",
                        stat
                    )
                })
                .get(stat),
            InnerArray::Viewed(v) => match stat {
                Stat::Max => {
                    let max = v.flatbuffer().stats()?.max();
                    max.and_then(|v| ScalarValue::try_from(v).ok())
                        .map(Precision::exact)
                }
                Stat::Min => {
                    let min = v.flatbuffer().stats()?.min();
                    min.and_then(|v| ScalarValue::try_from(v).ok().map(Precision::exact))
                }
                Stat::IsConstant => v.flatbuffer().stats()?.is_constant().map(Precision::exact),
                Stat::IsSorted => v.flatbuffer().stats()?.is_sorted().map(Precision::exact),
                Stat::IsStrictSorted => v
                    .flatbuffer()
                    .stats()?
                    .is_strict_sorted()
                    .map(Precision::exact),
                Stat::RunCount => v.flatbuffer().stats()?.run_count().map(Precision::exact),
                Stat::TrueCount => v.flatbuffer().stats()?.true_count().map(Precision::exact),
                Stat::NullCount => v.flatbuffer().stats()?.null_count().map(Precision::exact),
                Stat::BitWidthFreq => v
                    .flatbuffer()
                    .stats()?
                    .bit_width_freq()
                    .map(|v| v.iter().collect_vec())
                    .map(Precision::exact),
                Stat::TrailingZeroFreq => v
                    .flatbuffer()
                    .stats()?
                    .trailing_zero_freq()
                    .map(|v| v.iter().collect_vec())
                    .map(Precision::exact),
                Stat::UncompressedSizeInBytes => v
                    .flatbuffer()
                    .stats()?
                    .uncompressed_size_in_bytes()
                    .map(Precision::exact),
            },
        }
    }

    fn to_set(&self) -> StatsSet {
        match &self.0 {
            InnerArray::Owned(o) => o
                .stats_set
                .read()
                .unwrap_or_else(|_| vortex_panic!("Failed to acquire read lock on stats map"))
                .clone(),
            InnerArray::Viewed(_) => StatsSet::from_iter(
                all::<Stat>().filter_map(|stat| self.get(stat).map(|v| (stat, v.map(|v| v)))),
            ),
        }
    }

    fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        match &self.0 {
            InnerArray::Owned(o) => o
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
            InnerArray::Viewed(_) => {
                // We cannot modify stats on a view
            }
        }
    }

    fn clear(&self, stat: Stat) {
        match &self.0 {
            InnerArray::Owned(o) => {
                o.stats_set
                    .write()
                    .unwrap_or_else(|_| vortex_panic!("Failed to acquire write lock on stats map"))
                    .clear(stat);
            }
            InnerArray::Viewed(_) => {
                // We cannot modify stats on a view
            }
        }
    }

    fn compute(&self, stat: Stat) -> Option<ScalarValue> {
        self.compute_statistics(stat)
            .vortex_expect("compute_statistics must not fail")
            .get(stat)?
            .some_exact()
    }

    fn retain_only(&self, stats: &[Stat]) {
        match &self.0 {
            InnerArray::Owned(o) => {
                o.stats_set
                    .write()
                    .unwrap_or_else(|_| vortex_panic!("Failed to acquire write lock on stats map"))
                    .retain_only(stats);
            }
            InnerArray::Viewed(_) => {
                // We cannot modify stats on a view
            }
        }
    }
}
