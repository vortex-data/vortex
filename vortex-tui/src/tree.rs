use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::io::TokioFile;

use crate::TOKIO_RUNTIME;

pub fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let opened = TokioFile::open(file)?;

    let full = TOKIO_RUNTIME.block_on(async move {
        VortexOpenOptions::file(opened)
            .open()
            .await?
            .scan()
            .into_array()
            .await
    })?;

    println!("{}", full.tree_display());

    Ok(())
}
