//! Metadata accumulators track the per-chunk-of-a-column metadata, layout locations, and row counts.

use std::sync::Arc;

use itertools::Itertools;
use vortex_array::array::StructArray;
use vortex_array::builders::{builder_with_capacity, ArrayBuilder, ArrayBuilderExt};
use vortex_array::stats::{ArrayStatistics as _, Stat};
use vortex_array::validity::{ArrayValidity, Validity};
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

pub struct StatsAccumulator {
    stats: Vec<Stat>,
    builders: Vec<Box<dyn ArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: &DType, stats: Vec<Stat>) -> Self {
        let builders = stats
            .iter()
            .map(|s| {
                let dtype = match s {
                    Stat::BitWidthFreq => DType::List(
                        Arc::new(DType::Primitive(PType::U64, NonNullable)),
                        Nullable,
                    ),
                    Stat::TrailingZeroFreq => DType::List(
                        Arc::new(DType::Primitive(PType::U64, NonNullable)),
                        Nullable,
                    ),
                    Stat::IsConstant => DType::Bool(Nullable),
                    Stat::IsSorted => DType::Bool(Nullable),
                    Stat::IsStrictSorted => DType::Bool(Nullable),
                    Stat::Max => dtype.as_nullable(),
                    Stat::Min => dtype.as_nullable(),
                    Stat::RunCount => DType::Primitive(PType::U64, Nullable),
                    Stat::TrueCount => DType::Primitive(PType::U64, Nullable),
                    Stat::NullCount => DType::Primitive(PType::U64, Nullable),
                    Stat::UncompressedSizeInBytes => DType::Primitive(PType::U64, Nullable),
                };
                builder_with_capacity(&dtype, 1024)
            })
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

    pub fn into_array(mut self) -> VortexResult<Option<ArrayData>> {
        let mut names = vec![];
        let mut fields = vec![];

        for (stat, builder) in self.stats.iter().zip(self.builders.iter_mut()) {
            let values = builder.finish().map_err(|e| {
                e.with_context(format!("Failed to finish stat builder for {}", stat))
            })?;

            // We drop any all-null stats columns
            if values.logical_validity().null_count()? == values.len() {
                continue;
            }

            names.push(stat.to_string().into());
            fields.push(values);
        }

        if names.is_empty() {
            return Ok(None);
        }

        Ok(Some(
            StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)?
                .into_array(),
        ))
    }
}
