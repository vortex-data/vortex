use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::stream::ArrayStreamExt;

use crate::TOKIO_RUNTIME;

pub async fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let full = VortexOpenOptions::file()
        .open(file)
        .await?
        .scan()?
        .with_tokio_executor(TOKIO_RUNTIME.handle().clone())
        .into_array_stream()?
        .read_all()
        .await?;

    println!("{}", full.tree_display());

    Ok(())
}
