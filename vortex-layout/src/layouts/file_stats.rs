// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical as _;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::StatsSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::layouts::zoned::zone_map::StatsAccumulator;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

pub fn accumulate_stats(
    stream: SendableSequentialStream,
    stats: Arc<[Stat]>,
    max_variable_length_statistics_size: usize,
) -> (FileStatsAccumulator, SendableSequentialStream) {
    let accumulator =
        FileStatsAccumulator::new(stream.dtype(), stats, max_variable_length_statistics_size);
    let stream = SequentialStreamAdapter::new(
        stream.dtype().clone(),
        stream.scan(accumulator.clone(), |acc, item| {
            future::ready(Some(acc.process(item)))
        }),
    )
    .sendable();
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
        let accumulators = Arc::new(Mutex::new(match dtype.as_struct_fields_opt() {
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

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        if chunk.dtype().is_struct() {
            let chunk = chunk.to_struct();
            for (acc, field) in self
                .accumulators
                .lock()
                .iter_mut()
                .zip_eq(chunk.unmasked_fields().iter())
            {
                acc.push_chunk(field)?;
            }
        } else {
            self.accumulators.lock()[0].push_chunk(&chunk)?;
        }
        Ok((sequence_id, chunk))
    }

    pub fn stats_sets(&self) -> Vec<StatsSet> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        self.accumulators
            .lock()
            .iter_mut()
            .map(|acc| {
                acc.as_stats_table()
                    .vortex_expect("as_stats_table should not fail")
                    .map(|table| {
                        table
                            .to_stats_set(&self.stats, &mut ctx)
                            .vortex_expect("shouldn't fail to convert table we just created")
                    })
                    .unwrap_or_default()
            })
            .collect()
    }
}
