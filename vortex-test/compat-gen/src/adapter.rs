// Epoch C adapter — for Vortex v0.58.0 through HEAD
//
// Write: session.write_options(), returns WriteSummary, takes &mut sink
// Read:  session.open_options().open_buffer(buf) (sync), into_array_stream() (async)

use std::path::Path;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::file::{OpenOptionsSessionExt, WriteOptionsSessionExt};
use vortex::VortexSession;
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

    let session = VortexSession::default();
    runtime().block_on(async {
        let mut file = tokio::fs::File::create(path).await.map_err(|e| {
            vortex_error::vortex_err!("failed to create {}: {e}", path.display())
        })?;
        let _summary = session
            .write_options()
            .write(&mut file, stream)
            .await?;
        Ok(())
    })
}

/// Read a `.vortex` file from bytes, returning the arrays.
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    let session = VortexSession::default();
    let file = session.open_options().open_buffer(bytes)?;
    runtime().block_on(async {
        let arr = file
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?;
        Ok(vec![arr])
    })
}
