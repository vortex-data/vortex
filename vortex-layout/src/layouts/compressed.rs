// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::segments::SegmentSink;
use crate::{
    LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt as _, TaskExecutor, TaskExecutorExt as _,
};
use arcref::ArcRef;
use async_trait::async_trait;
use futures::{FutureExt as _, StreamExt as _};
use vortex_array::ArrayContext;
use vortex_array::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;

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

#[async_trait]
impl LayoutStrategy for BtrBlocksCompressedStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
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

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
            )
            .await
    }
}
