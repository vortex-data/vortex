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
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

fn runtime() -> VortexResult<Runtime> {
    Runtime::new().map_err(|e| vortex_error::vortex_err!("failed to create tokio runtime: {e}"))
}

/// Write a sequence of array chunks as a `.vortex` file with no compression.
///
/// Uses `FlatLayoutStrategy` directly — no repartitioning, no zone maps, no dictionary
/// encoding, no compression. Each chunk is serialized as a single flat segment.
pub fn write_file(path: &Path, chunk: ArrayRef) -> VortexResult<()> {
    let stream = ArrayStreamAdapter::new(chunk.dtype().clone(), stream::iter([Ok(chunk)]));

    let strategy: Arc<dyn vortex::layout::LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());

    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let mut file = tokio::fs::File::create(path)
            .await
            .map_err(|e| vortex_error::vortex_err!("failed to create {}: {e}", path.display()))?;
        let _summary = session
            .write_options()
            .with_strategy(strategy)
            .write(&mut file, stream)
            .await?;
        Ok(())
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
