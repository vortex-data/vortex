use std::future;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::SequentialArrayStream;
use crate::layouts::stats::stats_table::StatsAccumulator;
use crate::sequence::SequenceId;

pub fn accumulate_stats(
    dtype: &DType,
    stream: SequentialArrayStream,
    stats: Arc<[Stat]>,
) -> (FileStatsAccumulator, SequentialArrayStream) {
    let accumulator = FileStatsAccumulator::new(dtype, stats);
    let stream = Box::pin(stream.scan(accumulator.clone(), |acc, item| {
        future::ready(Some(acc.process(item)))
    }));
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
            Some(dtype) => dtype
                .fields()
                .map(|field_dtype| {
                    StatsAccumulator::new(&field_dtype, &stats, max_variable_length_statistics_size)
                })
                .collect(),
            None => [StatsAccumulator::new(dtype.clone(), &stats, max_variable_length_statistics_size)].into(),
        };

        Self {
            stats,
            accumulators,
        }
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        match chunk.as_struct_typed() {
            None => {
                self.accumulators.lock()[0].push_chunk(&chunk)?;
            }
            Some(array) => {
                for (acc, field) in self.accumulators.lock().iter_mut().zip_eq(array.fields()) {
                    acc.push_chunk(&field)?;
                }
            }
        }
        Ok((sequence_id, chunk))
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
