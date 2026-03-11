// Epoch A adapter — for Vortex v0.36.0
//
// Write: VortexWriteOptions::default(), returns sink W
// Read:  VortexOpenOptions::in_memory().open(buf).await (ASYNC)
// Scan:  into_array_stream() (async)

use std::path::Path;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex_array::stream::{ArrayStreamAdapter, ArrayStreamExt};
use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

fn runtime() -> Runtime {
    Runtime::new().expect("failed to create tokio runtime")
}

/// Write a sequence of array chunks as a `.vortex` file.
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));

    runtime().block_on(async {
        let file = tokio::fs::File::create(path).await.map_err(|e| {
            vortex_error::vortex_err!("failed to create {}: {e}", path.display())
        })?;
        // At 0.36.0: write() returns VortexResult<W> — we discard the sink.
        let _sink = VortexWriteOptions::default().write(file, stream).await?;
        Ok(())
    })
}

/// Read a `.vortex` file from bytes, returning the arrays.
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    runtime().block_on(async {
        let file = VortexOpenOptions::in_memory()
            .open(bytes) // async at 0.36.0
            .await?;
        let arr = file
            .scan()?
            .into_array_stream()? // async
            .read_all()
            .await?;
        Ok(vec![arr])
    })
}
