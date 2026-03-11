// Epoch B adapter — for Vortex v0.45.0 through v0.52.0
//
// Write: VortexWriteOptions::default(), returns sink W (same as Epoch A)
// Read:  VortexOpenOptions::in_memory().open(buf) (NOW SYNC)
// Scan:  into_array_iter() (sync iterator)

use std::path::Path;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex_array::iter::ArrayIteratorExt;
use vortex_array::stream::ArrayStreamAdapter;
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
        // At 0.45.0–0.52.0: same write API as Epoch A.
        let _sink = VortexWriteOptions::default().write(file, stream).await?;
        Ok(())
    })
}

/// Read a `.vortex` file from bytes, returning the arrays.
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    // No async runtime needed — both open and scan are sync at this epoch.
    let file = VortexOpenOptions::in_memory()
        .open(bytes)?; // sync at 0.45.0+
    let arr = file
        .scan()?
        .into_array_iter()? // sync iterator (replaced into_array_stream)
        .read_all()?; // sync read_all
    Ok(vec![arr])
}
