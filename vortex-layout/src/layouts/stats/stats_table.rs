use std::sync::Arc;

use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::builders::{ArrayBuilder, ArrayBuilderExt, builder_with_capacity};
use vortex_array::compute::sum;
use vortex_array::stats::{Precision, Stat, StatsSet};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

/// A table of statistics for a column.
/// Each row of the stats table corresponds to a chunk of the column.
///
/// Note that it's possible for the stats table to have no statistics.
#[derive(Clone)]
pub struct StatsTable {
    // The struct array backing the stats table
    array: StructArray,
    // The statistics that are included in the table.
    stats: Arc<[Stat]>,
}

impl StatsTable {
    /// Create StatsTable of given column_dtype from given array. Validates that the array matches expected
    /// structure for given list of stats
    pub fn try_new(
        column_dtype: DType,
        array: StructArray,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Self> {
        if &Self::dtype_for_stats_table(&column_dtype, &stats) != array.dtype() {
            vortex_bail!("Array dtype does not match expected stats table dtype");
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
            Arc::new(StructDType::from_iter(present_stats.iter().filter_map(
                |stat| {
                    stat.dtype(column_dtype)
                        .map(|dtype| (stat.name(), dtype.as_nullable()))
                },
            ))),
            Nullability::NonNullable,
        )
    }

    /// The struct array backing the stats table
    pub fn array(&self) -> &StructArray {
        &self.array
    }

    /// The statistics that are included in the table.
    pub fn present_stats(&self) -> &[Stat] {
        &self.stats
    }

    /// Return an aggregated stats set for the table.
    pub fn to_stats_set(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for stat in stats {
            let Some(array) = self.get_stat(*stat)? else {
                continue;
            };

            // Different stats need different aggregations
            match stat {
                // For stats that are associative, we can just compute them over the stat column
                Stat::Min | Stat::Max | Stat::Sum => {
                    if let Some(s) = array.statistics().compute_stat(*stat)? {
                        stats_set.set(*stat, Precision::exact(s))
                    }
                }
                // These stats sum up
                Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                    let sum = sum(&array)?
                        .cast(&DType::Primitive(PType::U64, Nullability::Nullable))?
                        .into_value();
                    stats_set.set(*stat, Precision::exact(sum));
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
/// TODO(ngates): we should make it such that the stats table stores a mirror of the DType
///  underneath each stats column. For example, `min: i32` for an `i32` array.
///  Or `min: {a: i32, b: i32}` for a struct array of type `{a: i32, b: i32}`.
///  See: <https://github.com/spiraldb/vortex/issues/1835>
pub struct StatsAccumulator {
    stats: Arc<[Stat]>,
    builders: Vec<Box<dyn ArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: DType, stats: &[Stat]) -> Self {
        let (stats, builders): (Vec<Stat>, _) = stats
            .iter()
            .filter_map(|s| {
                s.dtype(&dtype)
                    .map(|stat_dtype| (*s, builder_with_capacity(&stat_dtype.as_nullable(), 1024)))
            })
            .unzip();

        Self {
            stats: stats.into(),
            builders,
            length: 0,
        }
    }

    pub fn stats(&self) -> &[Stat] {
        &self.stats
    }

    pub fn push_chunk(&mut self, array: &dyn Array) -> VortexResult<()> {
        for (s, builder) in self.stats.iter().zip_eq(self.builders.iter_mut()) {
            if let Some(v) = array.statistics().compute_stat(*s)? {
                builder.append_scalar_value(v)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    /// Finishes the accumulator into a [`StatsTable`].
    ///
    /// Returns `None` if none of the requested statistics can be computed, for example they are
    /// not applicable to the column's data type.
    pub fn as_stats_table(&mut self) -> Option<StatsTable> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut stats = Vec::new();

        for (stat, builder) in self
            .stats
            .iter()
            .zip(self.builders.iter_mut())
            // We sort the stats so the DType is deterministic based on which stats are present.
            .sorted_unstable_by_key(|&(&s, _)| s)
        {
            let values = builder.finish();

            // We drop any all-null stats columns
            if values
                .invalid_count()
                .vortex_expect("failed to get invalid count")
                == values.len()
            {
                continue;
            }

            stats.push(*stat);
            names.push(stat.to_string().into());
            fields.push(values);
        }

        if names.is_empty() {
            return None;
        }

        Some(StatsTable {
            array: StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)
                .vortex_expect("Failed to create stats table"),
            stats: stats.into(),
        })
    }
}
