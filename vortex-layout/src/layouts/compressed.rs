// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arcref::ArcRef;
use futures::{FutureExt as _, StreamExt as _};
use vortex_array::ArrayContext;
use vortex_array::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;

use crate::scan::{TaskExecutor, TaskExecutorExt as _};
use crate::segments::SequenceWriter;
use crate::{
    LayoutStrategy, SendableLayoutWriter, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt as _,
};

/// A layout writer that compresses chunks using a sampling compressor.
pub struct BtrBlocksCompressedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    executor: Arc<dyn TaskExecutor>,
    parallelism: usize,
}

impl BtrBlocksCompressedStrategy {
    pub fn new(
        child: ArcRef<dyn LayoutStrategy>,
        executor: Arc<dyn TaskExecutor>,
        parallelism: usize,
    ) -> Self {
        Self {
            child,
            executor,
            parallelism,
        }
    }
}

impl LayoutStrategy for BtrBlocksCompressedStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        let executor = self.executor.clone();

        let dtype = stream.dtype().clone();
        let stream = stream
            .map(|chunk| {
                async {
                    let (sequence_id, chunk) = chunk?;
                    // Compute the stats for the chunk prior to compression
                    chunk
                        .statistics()
                        .compute_all(&Stat::all().collect::<Vec<_>>())?;
                    Ok((sequence_id, BtrBlocksCompressor.compress(&chunk)?))
                }
                .boxed()
            })
            .map(move |compress_future| executor.spawn(compress_future))
            .buffered(self.parallelism);

        self.child.write_stream(
            ctx,
            sequence_writer,
            SequentialStreamAdapter::new(dtype, stream).sendable(),
        )
    }
}
