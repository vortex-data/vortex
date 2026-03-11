// Epoch A adapter — for Vortex v0.36.0
//
// API at this version:
//   - VortexWriteOptions::default() (no session)
//   - .write(sink, stream).await returns VortexResult<W> (the sink back)
//   - ArrayStream must be Unpin

use std::path::Path;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::file::VortexWriteOptions;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

/// Write a sequence of array chunks as a `.vortex` file.
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));

    let rt = Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let file = tokio::fs::File::create(path).await.map_err(|e| {
            vortex_error::vortex_err!("failed to create {}: {e}", path.display())
        })?;
        // At 0.36.0, write() returns VortexResult<W> — we discard the sink.
        let _sink = VortexWriteOptions::default().write(file, stream).await?;
        Ok(())
    })
}
