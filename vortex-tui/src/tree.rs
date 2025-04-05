use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::io::TokioCloneFile;
use vortex::stream::ArrayStreamExt;

use crate::TOKIO_RUNTIME;

pub async fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let opened = TokioCloneFile::open(file)?;

    let full = VortexOpenOptions::file()
        .open(opened)
        .await?
        .scan()?
        .spawn_tokio(TOKIO_RUNTIME.handle().clone())?
        .read_all()
        .await?;

    println!("{}", full.tree_display());

    Ok(())
}
