// Epoch C adapter — for Vortex v0.58.0 through HEAD
//
// API changes from Epoch B:
//   - VortexWriteOptions no longer implements Default
//   - Must construct via VortexSession: session.write_options()
//   - .write(&mut sink, stream).await returns VortexResult<WriteSummary>
//   - WriteOptionsSessionExt trait provides session.write_options()

use std::path::Path;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::file::WriteOptionsSessionExt;
use vortex::VortexSession;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

/// Write a sequence of array chunks as a `.vortex` file.
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));

    let session = VortexSession::default();
    let rt = Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        let mut file = tokio::fs::File::create(path).await.map_err(|e| {
            vortex_error::vortex_err!("failed to create {}: {e}", path.display())
        })?;
        // At 0.58.0+: write() returns WriteSummary, takes &mut sink.
        let _summary = session
            .write_options()
            .write(&mut file, stream)
            .await?;
        Ok(())
    })
}
