use std::path::Path;

use vortex::error::VortexResult;
use vortex::session::VortexSession;
use vortex::stream::ArrayStreamExt;

use crate::TOKIO_RUNTIME;

pub async fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let session = VortexSession::default();
    let full = session
        .open_blocking(file)?
        .scan()?
        .with_tokio_executor(TOKIO_RUNTIME.handle().clone())
        .into_array_stream()?
        .read_all()
        .await?;

    println!("{}", full.tree_display());

    Ok(())
}
