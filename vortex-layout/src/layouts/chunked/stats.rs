//! Metadata accumulators track the per-chunk-of-a-column metadata, layout locations, and row counts.

use itertools::Itertools;
use vortex_array::array::StructArray;
use vortex_array::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt};
use vortex_array::stats::{ArrayStatistics as _, Stat};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;

pub struct StatsAccumulator {
    stats: Vec<Stat>,
    builders: Vec<Box<dyn ArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: &DType, mut stats: Vec<Stat>) -> Self {
        // Sort stats by their ordinal so we can recreate their dtype from bitset
        stats.sort_by_key(|s| u8::from(*s));
        let builders = stats
            .iter()
            .map(|s| builder_with_capacity(&s.dtype(dtype).as_nullable(), 1024))
            .collect();
        Self {
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

    pub fn as_array(&mut self) -> VortexResult<Option<StatsArray>> {
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

        Ok(Some(StatsArray(
            StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)?
                .into_array(),
            stats,
        )))
    }
}

pub struct StatsArray(pub ArrayData, pub Vec<Stat>);
