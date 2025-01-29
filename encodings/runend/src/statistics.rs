use std::cmp;

use arrow_buffer::BooleanBuffer;
use itertools::Itertools;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::StatisticsVTable;
use vortex_array::IntoArrayVariant as _;
use vortex_dtype::{match_each_unsigned_integer_ptype, DType, NativePType};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::ScalarValue;

use crate::{RunEndArray, RunEndEncoding};

impl StatisticsVTable<RunEndArray> for RunEndEncoding {
    fn compute_statistics(&self, array: &RunEndArray, stat: Stat) -> VortexResult<StatsSet> {
        let maybe_stat = match stat {
            Stat::Min | Stat::Max => array.values().statistics().compute(stat),
            Stat::IsSorted => Some(ScalarValue::from(
                array
                    .values()
                    .statistics()
                    .compute_is_sorted()
                    .unwrap_or(false)
                    && array.logical_validity()?.all_true(),
            )),
            Stat::TrueCount => match array.dtype() {
                DType::Bool(_) => Some(ScalarValue::from(array.true_count()?)),
                _ => None,
            },
            Stat::NullCount => Some(ScalarValue::from(array.null_count()?)),
            _ => None,
        };

        let mut stats = StatsSet::default();
        if let Some(stat_value) = maybe_stat {
            stats.set(stat, stat_value);
        }
        Ok(stats)
    }
}

impl RunEndArray {
    fn true_count(&self) -> VortexResult<u64> {
        let ends = self.ends().into_primitive()?;
        let values = self.values().into_bool()?.boolean_buffer();

        match_each_unsigned_integer_ptype!(ends.ptype(), |$P| self.typed_true_count(ends.as_slice::<$P>(), values))
    }

    fn typed_true_count<P: NativePType + Into<u64>>(
        &self,
        decompressed_ends: &[P],
        decompressed_values: BooleanBuffer,
    ) -> VortexResult<u64> {
        Ok(match self.values().logical_validity()? {
            Mask::AllTrue(_) => {
                let mut begin = self.offset() as u64;
                decompressed_ends
                    .iter()
                    .copied()
                    .zip_eq(&decompressed_values)
                    .map(|(end, bool_value)| {
                        let end: u64 = end.into();
                        let len = end - begin;
                        begin = end;
                        len * u64::from(bool_value)
                    })
                    .sum()
            }
            Mask::AllFalse(_) => 0,
            Mask::Values(values) => {
                let mut is_valid = values.indices().iter();
                match is_valid.next() {
                    None => self.len() as u64,
                    Some(&valid_index) => {
                        let mut true_count: u64 = 0;
                        let offsetted_begin = self.offset() as u64;
                        let offsetted_len = (self.len() + self.offset()) as u64;
                        let begin = if valid_index == 0 {
                            offsetted_begin
                        } else {
                            decompressed_ends[valid_index - 1].into()
                        };

                        let end = cmp::min(decompressed_ends[valid_index].into(), offsetted_len);
                        true_count += decompressed_values.value(valid_index) as u64 * (end - begin);

                        for &valid_index in is_valid {
                            let valid_end: u64 = decompressed_ends[valid_index].into();
                            let end = cmp::min(valid_end, offsetted_len);
                            true_count +=
                                decompressed_values.value(valid_index) as u64 * (end - valid_end);
                        }

                        true_count
                    }
                }
            }
        })
    }

    fn null_count(&self) -> VortexResult<u64> {
        let ends = self.ends().into_primitive()?;
        let null_count = match self.values().logical_validity()? {
            Mask::AllTrue(_) => 0u64,
            Mask::AllFalse(_) => self.len() as u64,
            Mask::Values(mask) => {
                match_each_unsigned_integer_ptype!(ends.ptype(), |$P| self.null_count_with_array_validity(ends.as_slice::<$P>(), mask.boolean_buffer()))
            }
        };
        Ok(null_count)
    }

    fn null_count_with_array_validity<P: NativePType + Into<u64>>(
        &self,
        decompressed_ends: &[P],
        is_valid: &BooleanBuffer,
    ) -> u64 {
        let mut is_valid = is_valid.set_indices();
        match is_valid.next() {
            None => self.len() as u64,
            Some(valid_index) => {
                let offsetted_len = (self.len() + self.offset()) as u64;
                let mut null_count: u64 = self.len() as u64;
                let begin = if valid_index == 0 {
                    0
                } else {
                    decompressed_ends[valid_index - 1].into()
                };

                let end = cmp::min(decompressed_ends[valid_index].into(), offsetted_len);
                null_count -= end - begin;

                for valid_index in is_valid {
                    let end = cmp::min(decompressed_ends[valid_index].into(), offsetted_len);
                    null_count -= end - decompressed_ends[valid_index - 1].into();
                }

                null_count
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::array::BoolArray;
    use vortex_array::compute::slice;
    use vortex_array::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayData;
    use vortex_buffer::buffer;

    use crate::RunEndArray;

    #[test]
    fn test_runend_int_stats() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        assert_eq!(arr.statistics().compute_as::<i32>(Stat::Min).unwrap(), 1);
        assert_eq!(arr.statistics().compute_as::<i32>(Stat::Max).unwrap(), 3);
        assert_eq!(
            arr.statistics().compute_as::<u64>(Stat::NullCount).unwrap(),
            0
        );
        assert!(arr.statistics().compute_as::<bool>(Stat::IsSorted).unwrap());
    }

    #[test]
    fn test_runend_bool_stats() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            BoolArray::try_new(
                BooleanBuffer::from_iter([true, true, false]),
                Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            )
            .unwrap()
            .into_array(),
        )
        .unwrap();

        assert!(!arr.statistics().compute_as::<bool>(Stat::Min).unwrap());
        assert!(arr.statistics().compute_as::<bool>(Stat::Max).unwrap());
        assert_eq!(
            arr.statistics().compute_as::<u64>(Stat::NullCount).unwrap(),
            3
        );
        assert!(!arr.statistics().compute_as::<bool>(Stat::IsSorted).unwrap());
        assert_eq!(
            arr.statistics().compute_as::<u64>(Stat::TrueCount).unwrap(),
            2
        );

        let sliced = slice(arr, 4, 7).unwrap();

        assert!(!sliced.statistics().compute_as::<bool>(Stat::Min).unwrap());
        assert!(!sliced.statistics().compute_as::<bool>(Stat::Max).unwrap());
        assert_eq!(
            sliced
                .statistics()
                .compute_as::<u64>(Stat::NullCount)
                .unwrap(),
            1
        );
        // Not sorted because null must come last
        assert!(!sliced
            .statistics()
            .compute_as::<bool>(Stat::IsSorted)
            .unwrap());
        assert_eq!(
            sliced
                .statistics()
                .compute_as::<u64>(Stat::TrueCount)
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_all_invalid_true_count() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            BoolArray::from_iter([None, None, None]).into_array(),
        )
        .unwrap()
        .into_array();
        assert_eq!(
            arr.statistics().compute_as::<u64>(Stat::TrueCount).unwrap(),
            0
        );
        assert_eq!(
            arr.statistics().compute_as::<u64>(Stat::NullCount).unwrap(),
            10
        );
    }
}
