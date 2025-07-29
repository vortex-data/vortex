// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{Array, ToCanonical as _};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};

use crate::layouts::zoned::zone_map::StatsAccumulator;

pub fn accumulate_stats<S: ArrayStream>(
    stream: S,
    stats: Arc<[Stat]>,
    max_variable_length_statistics_size: usize,
) -> (FileStatsAccumulator, impl ArrayStream) {
    let accumulator =
        FileStatsAccumulator::new(stream.dtype(), stats, max_variable_length_statistics_size);
    let accumulator2 = accumulator.clone();

    let stream = ArrayStreamAdapter::new(
        stream.dtype().clone(),
        stream.map(move |chunk| {
            let accumulator = accumulator2.clone();
            chunk.and_then(move |c| {
                accumulator.process(&c)?;
                Ok(c)
            })
        }),
    );

    (accumulator, stream)
}

/// An array stream processor that computes aggregate statistics for all fields.
///
/// Note: for now this only collects top-level struct fields.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    stats: Arc<[Stat]>,
    accumulators: Arc<Mutex<Vec<StatsAccumulator>>>,
}

impl FileStatsAccumulator {
    fn new(dtype: &DType, stats: Arc<[Stat]>, max_variable_length_statistics_size: usize) -> Self {
        let accumulators = Arc::new(Mutex::new(match dtype.as_struct() {
            Some(struct_dtype) => {
                if dtype.nullability() == Nullability::Nullable {
                    // top level dtype could be nullable, but we don't support it yet
                    vortex_panic!(
                        "FileStatsAccumulator temporarily does not support nullable top-level structs, got: {}. Use Validity::NonNullable",
                        dtype
                    );
                }

                struct_dtype
                    .fields()
                    .map(|field_dtype| {
                        StatsAccumulator::new(
                            &field_dtype,
                            &stats,
                            max_variable_length_statistics_size,
                        )
                    })
                    .collect()
            }
            None => [StatsAccumulator::new(
                dtype,
                &stats,
                max_variable_length_statistics_size,
            )]
            .into(),
        }));

        Self {
            stats,
            accumulators,
        }
    }

    fn process(&self, chunk: &dyn Array) -> VortexResult<()> {
        if chunk.dtype().is_struct() {
            let chunk = chunk.to_struct()?;
            for (acc, field) in self.accumulators.lock().iter_mut().zip_eq(chunk.fields()) {
                acc.push_chunk(field)?;
            }
            Ok(())
        } else {
            self.accumulators.lock()[0].push_chunk(chunk)
        }
    }

    pub fn stats_sets(&self) -> Vec<StatsSet> {
        self.accumulators
            .lock()
            .iter_mut()
            .map(|acc| {
                acc.as_stats_table()
                    .map(|table| {
                        table
                            .to_stats_set(&self.stats)
                            .vortex_expect("shouldn't fail to convert table we just created")
                    })
                    .unwrap_or_default()
            })
            .collect()
    }
}
