// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Epoch C adapter — for Vortex v0.58.0 through HEAD
//
// Write: session.write_options(), returns WriteSummary, takes &mut sink
// Read:  session.open_options().open_buffer(buf) (sync), into_array_stream() (async)

use std::path::Path;
use std::sync::Arc;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_array::ArrayRef;
use vortex_array::ArrayVisitorExt;
use vortex_array::DynArray;
use vortex_array::expr::stats::Stat;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

fn runtime() -> VortexResult<Runtime> {
    Runtime::new().map_err(|e| vortex_err!("failed to create tokio runtime: {e}"))
}

/// Compute all statistics on every node in the array tree.
///
/// The flat layout writer does not compute stats itself — it only serializes stats already
/// cached on each array node. This function walks the entire tree and forces computation of
/// all stats so they are present in the serialized output.
pub fn compute_all_stats(array: &ArrayRef) -> VortexResult<()> {
    let all_stats: Vec<Stat> = Stat::all().collect();
    for node in array.depth_first_traversal() {
        let computed = node.statistics().compute_all(&all_stats)?;
        node.statistics().set_iter(computed.into_iter());
    }
    Ok(())
}

/// Write a sequence of array chunks as a `.vortex` file with no compression.
///
/// Uses `FlatLayoutStrategy` directly — no repartitioning, no zone maps, no dictionary
/// encoding, no compression. Each chunk is serialized as a single flat segment.
pub fn write_file(path: &Path, chunk: ArrayRef) -> VortexResult<()> {
    write_compressed(path, chunk, Arc::new(FlatLayoutStrategy::default()))
}

/// Write a sequence of array chunks to an in-memory `.vortex` byte buffer with no compression.
pub fn write_file_to_bytes(chunk: ArrayRef) -> VortexResult<ByteBuffer> {
    write_compressed_to_bytes(chunk, Arc::new(FlatLayoutStrategy::default()))
}

/// Write a `.vortex` file using a caller-provided layout strategy (compressor pipeline).
pub fn write_compressed(
    path: &Path,
    chunk: ArrayRef,
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<()> {
    let stream = ArrayStreamAdapter::new(chunk.dtype().clone(), stream::iter([Ok(chunk)]));

    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let mut file = tokio::fs::File::create(path)
            .await
            .map_err(|e| vortex_err!("failed to create {}: {e}", path.display()))?;
        let _summary = session
            .write_options()
            .with_strategy(strategy)
            .write(&mut file, stream)
            .await?;
        Ok(())
    })
}

/// Write a `.vortex` file into an in-memory byte buffer using a caller-provided layout strategy.
pub fn write_compressed_to_bytes(
    chunk: ArrayRef,
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<ByteBuffer> {
    let stream = ArrayStreamAdapter::new(chunk.dtype().clone(), stream::iter([Ok(chunk)]));

    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let mut bytes = Vec::new();
        let _summary = session
            .write_options()
            .with_strategy(strategy)
            .write(&mut bytes, stream)
            .await?;
        Ok(ByteBuffer::from(bytes))
    })
}

/// Read a `.vortex` file from bytes, returning the arrays.
pub fn read_file(bytes: ByteBuffer) -> VortexResult<ArrayRef> {
    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let file = session.open_options().open_buffer(bytes)?;
        file.scan()?.into_array_stream()?.read_all().await
    })
}
