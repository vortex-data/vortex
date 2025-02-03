use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::{ExecutionMode, Scan, SplitBy, VortexOpenOptions};
use vortex::io::TokioFile;
use vortex::sampling_compressor::ALL_ENCODINGS_CONTEXT;
use vortex::stream::ArrayStreamExt;

use crate::TOKIO_RUNTIME;

pub fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let opened = TokioFile::open(file)?;

    let full = TOKIO_RUNTIME.block_on(async move {
        let file = VortexOpenOptions::new(ALL_ENCODINGS_CONTEXT.clone())
            .with_execution_mode(ExecutionMode::Inline)
            .with_split_by(SplitBy::Layout)
            .open(opened)
            .await?;

        // TODO(aduffy): scan with paging.
        file.scan(Scan::all())?.into_array_data().await
    })?;

    println!("{}", full.tree_display());

    Ok(())
}
