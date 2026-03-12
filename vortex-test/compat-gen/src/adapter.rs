// Epoch C adapter — for Vortex v0.58.0 through HEAD
//
// Write: session.write_options(), returns WriteSummary, takes &mut sink
// Read:  session.open_options().open_buffer(buf) (sync), into_array_stream() (async)

use std::path::Path;

use futures::stream;
use tokio::runtime::Runtime;
use vortex::VortexSessionDefault;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex_array::ArrayRef;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

fn runtime() -> VortexResult<Runtime> {
    Runtime::new().map_err(|e| vortex_error::vortex_err!("failed to create tokio runtime: {e}"))
}

/// Write a sequence of array chunks as a `.vortex` file.
#[allow(dead_code)]
pub fn write_file(path: &Path, chunks: Vec<ArrayRef>) -> VortexResult<()> {
    let dtype = chunks[0].dtype().clone();
    let stream = ArrayStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)));

    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let mut file = tokio::fs::File::create(path)
            .await
            .map_err(|e| vortex_error::vortex_err!("failed to create {}: {e}", path.display()))?;
        let _summary = session.write_options().write(&mut file, stream).await?;
        Ok(())
    })
}

/// Read a `.vortex` file from bytes, returning the arrays.
#[allow(dead_code)]
pub fn read_file(bytes: ByteBuffer) -> VortexResult<Vec<ArrayRef>> {
    runtime()?.block_on(async {
        let session = VortexSession::default().with_tokio();
        let file = session.open_options().open_buffer(bytes)?;
        let arr = file.scan()?.into_array_stream()?.read_all().await?;
        Ok(vec![arr])
    })
}
