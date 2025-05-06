use std::sync::Arc;

use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::compute::sum;
use vortex_array::stats::{Precision, Stat, StatsSet};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::layouts::zoned::builder::{
    MAX_IS_TRUNCATED, MIN_IS_TRUNCATED, StatsArrayBuilder, stats_builder_with_capacity,
};

/// A zone map containing statistics for a column.
/// Each row of the zone map corresponds to a chunk of the column.
///
/// Note that it's possible for the zone map to have no statistics.
#[derive(Clone)]
pub struct ZoneMap {
    // The struct array backing the zone map
    array: StructArray,
    // The statistics that are included in the table.
    stats: Arc<[Stat]>,
}

impl ZoneMap {
    /// Create StatsTable of given column_dtype from given array. Validates that the array matches expected
    /// structure for given list of stats
    pub fn try_new(
        column_dtype: DType,
        array: StructArray,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Self> {
        if &Self::dtype_for_stats_table(&column_dtype, &stats) != array.dtype() {
            vortex_bail!("Array dtype does not match expected zone map dtype");
        }
        Ok(Self::unchecked_new(array, stats))
    }

    /// Create StatsTable without validating return array against expected stats
    pub fn unchecked_new(array: StructArray, stats: Arc<[Stat]>) -> Self {
        Self { array, stats }
    }

    /// Returns the DType of the statistics table given a set of statistics and column [`DType`].
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        assert!(present_stats.is_sorted(), "Stats must be sorted");
        DType::Struct(
            Arc::new(StructDType::from_iter(
                present_stats
                    .iter()
                    .filter_map(|stat| {
                        stat.dtype(column_dtype)
                            .map(|dtype| (stat, dtype.as_nullable()))
                    })
                    .flat_map(|(s, dt)| match s {
                        Stat::Max => vec![
                            (s.name(), dt),
                            (MAX_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
                        ],
                        Stat::Min => vec![
                            (s.name(), dt),
                            (MIN_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
                        ],
                        _ => vec![(s.name(), dt)],
                    }),
            )),
            Nullability::NonNullable,
        )
    }

    /// The struct array backing the zone map
    pub fn array(&self) -> &StructArray {
        &self.array
    }

    /// The statistics that are included in the table.
    pub fn present_stats(&self) -> &Arc<[Stat]> {
        &self.stats
    }

    /// Return an aggregated stats set for the table.
    pub fn to_stats_set(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for &stat in stats {
            let Some(array) = self.get_stat(stat)? else {
                continue;
            };

            // Different stats need different aggregations
            match stat {
                // For stats that are associative, we can just compute them over the stat column
                Stat::Min | Stat::Max | Stat::Sum => {
                    if let Some(s) = array.statistics().compute_stat(stat)? {
                        stats_set.set(stat, Precision::exact(s))
                    }
                }
                // These stats sum up
                Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                    let sum = sum(&array)?
                        .cast(&DType::Primitive(PType::U64, Nullability::Nullable))?
                        .into_value();
                    stats_set.set(stat, Precision::exact(sum));
                }
                // We could implement these aggregations in the future, but for now they're unused
                Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {}
            }
        }
        Ok(stats_set)
    }

    /// Return the array for a given stat.
    pub fn get_stat(&self, stat: Stat) -> VortexResult<Option<ArrayRef>> {
        Ok(self.array.field_by_name_opt(stat.name()).cloned())
    }
}

/// Accumulates statistics for a column.
///
/// TODO(ngates): we should make it such that the zone map stores a mirror of the DType
///  underneath each stats column. For example, `min: i32` for an `i32` array.
///  Or `min: {a: i32, b: i32}` for a struct array of type `{a: i32, b: i32}`.
///  See: <https://github.com/vortex-data/vortex/issues/1835>
pub struct StatsAccumulator {
    builders: Vec<Box<dyn StatsArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: &DType, stats: &[Stat], max_variable_length_statistics_size: usize) -> Self {
        let builders = stats
            .iter()
            .filter_map(|&s| {
                s.dtype(dtype).map(|stat_dtype| {
                    stats_builder_with_capacity(
                        s,
                        &stat_dtype.as_nullable(),
                        1024,
                        max_variable_length_statistics_size,
                    )
                })
            })
            .collect::<Vec<_>>();

        Self {
            builders,
            length: 0,
        }
    }

    pub fn push_chunk(&mut self, array: &dyn Array) -> VortexResult<()> {
        for builder in self.builders.iter_mut() {
            if let Some(v) = array.statistics().compute_stat(builder.stat())? {
                builder.append_scalar_value(v)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    /// Finishes the accumulator into a [`ZoneMap`].
    ///
    /// Returns `None` if none of the requested statistics can be computed, for example they are
    /// not applicable to the column's data type.
    pub fn as_stats_table(&mut self) -> Option<ZoneMap> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut stats = Vec::new();

        for builder in self
            .builders
            .iter_mut()
            // We sort the stats so the DType is deterministic based on which stats are present.
            .sorted_unstable_by_key(|b| b.stat())
        {
            let values = builder.finish();

            // We drop any all-null stats columns
            if values
                .all_invalid()
                .vortex_expect("failed to get invalid count")
            {
                continue;
            }

            stats.push(builder.stat());
            names.extend(values.names);
            fields.extend(values.arrays);
        }

        if names.is_empty() {
            return None;
        }

        Some(ZoneMap {
            array: StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)
                .vortex_expect("Failed to create zone map"),
            stats: stats.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rstest::rstest;
    use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
    use vortex_array::stats::Stat;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_error::{VortexExpect, VortexUnwrap};

    use crate::layouts::zoned::zone_map::StatsAccumulator;
    use crate::layouts::zoned::{MAX_IS_TRUNCATED, MIN_IS_TRUNCATED};

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncates_accumulated_stats(#[case] dtype: DType) {
        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), 2);
        builder.append_value("Value to be truncated");
        builder.append_value("untruncated");
        let mut builder2 = VarBinViewBuilder::with_capacity(dtype, 2);
        builder2.append_value("Another");
        builder2.append_value("wait a minute");
        let mut acc =
            StatsAccumulator::new(builder.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&builder.finish()).vortex_unwrap();
        acc.push_chunk(&builder2.finish()).vortex_unwrap();
        let stats_table = acc.as_stats_table().vortex_expect("Must have stats table");
        assert_eq!(
            stats_table.array.names().as_ref(),
            &[
                Stat::Max.name().into(),
                MAX_IS_TRUNCATED.into(),
                Stat::Min.name().into(),
                MIN_IS_TRUNCATED.into(),
            ]
        );
        assert_eq!(
            stats_table.array.fields()[1]
                .to_bool()
                .vortex_unwrap()
                .boolean_buffer(),
            &BooleanBuffer::from(vec![false, true])
        );
        assert_eq!(
            stats_table.array.fields()[3]
                .to_bool()
                .vortex_unwrap()
                .boolean_buffer(),
            &BooleanBuffer::from(vec![true, false])
        );
    }

    #[test]
    fn always_adds_is_truncated_column() {
        let array = buffer![0, 1, 2].into_array();
        let mut acc = StatsAccumulator::new(array.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&array).vortex_unwrap();
        let stats_table = acc.as_stats_table().vortex_expect("Must have stats table");
        assert_eq!(
            stats_table.array.names().as_ref(),
            &[
                Stat::Max.name().into(),
                MAX_IS_TRUNCATED.into(),
                Stat::Min.name().into(),
                MIN_IS_TRUNCATED.into(),
                Stat::Sum.name().into(),
            ]
        );
        assert_eq!(
            stats_table.array.fields()[1]
                .to_bool()
                .vortex_unwrap()
                .boolean_buffer(),
            &BooleanBuffer::from(vec![false])
        );
        assert_eq!(
            stats_table.array.fields()[3]
                .to_bool()
                .vortex_unwrap()
                .boolean_buffer(),
            &BooleanBuffer::from(vec![false])
        );
    }
}
