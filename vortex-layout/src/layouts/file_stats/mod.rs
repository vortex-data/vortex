// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Accumulate finalized file summaries while the input stream is written.
//!
//! The stream adapter feeds chunks to each top-level field in sequence order. Field accumulators
//! own their states for the duration of one write. Arrays may retain finalized cached results.

mod field;

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::stats::AggregateResults;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use self::field::FieldAccumulator;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Accumulate file summaries as the stream is consumed, retaining its sequence order.
///
/// Unsupported field types have no result. A top-level struct must be non-nullable because its
/// fields are accumulated independently of struct validity.
pub fn accumulate_stats(
    stream: SendableSequentialStream,
    aggregates: Arc<[AggregateFnRef]>,
    max_variable_length_statistics_size: usize,
    session: &VortexSession,
) -> VortexResult<(FileStatsAccumulator, SendableSequentialStream)> {
    let accumulator = FileStatsAccumulator::new(
        stream.dtype(),
        &aggregates,
        max_variable_length_statistics_size,
        session,
    )?;
    let stream = SequentialStreamAdapter::new(
        stream.dtype().clone(),
        stream.scan(accumulator.clone(), |acc, item| {
            future::ready(Some(acc.process(item)))
        }),
    )
    .sendable();

    Ok((accumulator, stream))
}

/// Collect summaries for the file's top-level fields, or for its single non-struct column.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    accumulators: Arc<Mutex<Vec<FieldAccumulator>>>,
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl FileStatsAccumulator {
    fn new(
        dtype: &DType,
        aggregates: &[AggregateFnRef],
        max_length: usize,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let accumulators = match dtype.as_struct_fields_opt() {
            Some(fields) => {
                vortex_ensure!(
                    dtype.nullability() == Nullability::NonNullable,
                    "File summaries require a non-nullable top-level struct, got {dtype}"
                );
                fields
                    .fields()
                    .map(|dtype| FieldAccumulator::new(&dtype, aggregates, max_length))
                    .collect::<VortexResult<Vec<_>>>()?
            }
            None => vec![FieldAccumulator::new(dtype, aggregates, max_length)?],
        };

        Ok(Self {
            accumulators: Arc::new(Mutex::new(accumulators)),
            ctx: Arc::new(Mutex::new(session.create_execution_ctx())),
        })
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        let mut ctx = self.ctx.lock();

        if chunk.dtype().is_struct() {
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            for (acc, field) in self
                .accumulators
                .lock()
                .iter_mut()
                .zip_eq(struct_chunk.iter_unmasked_fields())
            {
                acc.push_chunk(field, &mut ctx)?;
            }
        } else {
            self.accumulators.lock()[0].push_chunk(&chunk, &mut ctx)?;
        }

        Ok((sequence_id, chunk))
    }

    /// Finalized summaries with string and binary bounds truncated to the configured limit.
    ///
    /// Call after consuming the stream to obtain complete file summaries. A stream with no chunks
    /// produces an empty collection for each field. Received empty chunks preserve supported count,
    /// sum, and physical-size results while omitting extrema and flags. An observed Sum overflow
    /// is retained as an exact null result, which is distinct from a missing result.
    pub fn results(&self) -> VortexResult<Vec<AggregateResults>> {
        self.accumulators
            .lock()
            .iter()
            .map(FieldAccumulator::results)
            .collect()
    }
}

#[cfg(test)]
mod tests;
