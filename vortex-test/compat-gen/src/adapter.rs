// Epoch B adapter — for Vortex v0.45.0 through v0.52.0
//
// API changes from Epoch A:
//   - VortexWriteOptions::default() still works (no session)
//   - .write(sink, stream).await still returns VortexResult<W>
//   - Stream now requires Send + 'static (not just Unpin)
//   - Also has .write_blocking(sink, stream) -> VortexResult<W>

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
        // At 0.45.0–0.52.0: same as Epoch A, write() returns VortexResult<W>.
        // Stream bound changed to `S: ArrayStream + Unpin + Send + 'static`.
        let _sink = VortexWriteOptions::default().write(file, stream).await?;
        Ok(())
    })
}
