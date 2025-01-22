use std::sync::Arc;

use itertools::Itertools;
use vortex_array::array::StructArray;
use vortex_array::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt};
use vortex_array::compute::try_cast;
use vortex_array::stats::{ArrayStatistics as _, Stat, StatsSet};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{DType, Nullability, PType, StructDType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

/// A table of statistics for a column.
/// Each row of the stats table corresponds to a chunk of the column.
///
/// Note that it's possible for the stats table to have no statistics.
#[derive(Clone)]
pub struct StatsTable {
    // The DType of the column for which these stats are computed.
    column_dtype: DType,
    // The struct array backing the stats table
    array: ArrayData,
    // The statistics that are included in the table.
    stats: Arc<[Stat]>,
}

impl StatsTable {
    pub fn try_new(
        column_dtype: DType,
        array: ArrayData,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Self> {
        if &Self::dtype_for_stats_table(&column_dtype, &stats) != array.dtype() {
            vortex_bail!("Array dtype does not match expected stats table dtype");
        }
        Ok(Self {
            column_dtype,
            array,
            stats,
        })
    }

    /// Returns the DType of the statistics table given a set of statistics and column [`DType`].
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        DType::Struct(
            Arc::new(StructDType::from_iter(present_stats.iter().map(|stat| {
                (stat.name(), stat.dtype(column_dtype).as_nullable())
            }))),
            Nullability::NonNullable,
        )
    }

    /// The DType of the column for which these stats are computed.
    pub fn column_dtype(&self) -> &DType {
        &self.column_dtype
    }

    /// The struct array backing the stats table
    pub fn array(&self) -> &ArrayData {
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
                Stat::Min | Stat::Max => {
                    if let Some(s) = array.statistics().compute(*stat) {
                        stats_set.set(*stat, s)
                    }
                }
                // These stats sum up
                Stat::TrueCount | Stat::NullCount | Stat::UncompressedSizeInBytes => {
                    // TODO(ngates): use Stat::Sum when we add it.
                    let parray =
                        try_cast(array, &DType::Primitive(PType::U64, Nullability::Nullable))?
                            .into_primitive()?;
                    let sum: u64 = parray
                        .as_slice::<u64>()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, v)| parray.validity().is_valid(i).then_some(*v))
                        .sum();
                    stats_set.set(*stat, sum);
                }
                // We could implement these aggregations in the future, but for now they're unused
                Stat::BitWidthFreq
                | Stat::TrailingZeroFreq
                | Stat::RunCount
                | Stat::IsConstant
                | Stat::IsSorted
                | Stat::IsStrictSorted => {}
            }
        }
        Ok(stats_set)
    }

    /// Return the array for a given stat.
    pub fn get_stat(&self, stat: Stat) -> VortexResult<Option<ArrayData>> {
        Ok(self
            .array
            .as_struct_array()
            .vortex_expect("Stats table must be a struct array")
            .maybe_null_field_by_name(stat.name()))
    }
}

/// Accumulates statistics for a column.
///
/// TODO(ngates): we should make it such that the stats table stores a mirror of the DType
///  underneath each stats column. For example, `min: i32` for an `i32` array.
///  Or `min: {a: i32, b: i32}` for a struct array of type `{a: i32, b: i32}`.
///  See: <https://github.com/spiraldb/vortex/issues/1835>
pub struct StatsAccumulator {
    column_dtype: DType,
    stats: Vec<Stat>,
    builders: Vec<Box<dyn ArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: DType, mut stats: Vec<Stat>) -> Self {
        // Sort stats by their ordinal so we can recreate their dtype from bitset
        stats.sort_by_key(|s| u8::from(*s));
        let builders = stats
            .iter()
            .map(|s| builder_with_capacity(&s.dtype(&dtype).as_nullable(), 1024))
            .collect();
        Self {
            column_dtype: dtype,
            stats,
            builders,
            length: 0,
        }
    }

    pub fn push_chunk(&mut self, array: &ArrayData) -> VortexResult<()> {
        for (s, builder) in self.stats.iter().zip_eq(self.builders.iter_mut()) {
            if let Some(v) = array.statistics().compute(*s) {
                builder.append_scalar(&v.cast(builder.dtype())?)?;
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
    pub fn as_stats_table(&mut self) -> VortexResult<Option<StatsTable>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut stats = Vec::new();

        for (stat, builder) in self.stats.iter().zip(self.builders.iter_mut()) {
            let values = builder
                .finish()
                .map_err(|e| e.with_context(format!("Failed to finish stat builder for {stat}")))?;

            // We drop any all-null stats columns
            if values.logical_validity().null_count()? == values.len() {
                continue;
            }

            stats.push(*stat);
            names.push(stat.to_string().into());
            fields.push(values);
        }

        if names.is_empty() {
            return Ok(None);
        }

        Ok(Some(StatsTable {
            column_dtype: self.column_dtype.clone(),
            array: StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)?
                .into_array(),
            stats: stats.into(),
        }))
    }
}
