use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::io::TokioFile;

use crate::TOKIO_RUNTIME;

pub fn segments(file: impl AsRef<Path>) -> VortexResult<()> {
    let opened = TokioFile::open(file)?;

    let vxf = TOKIO_RUNTIME.block_on(async move { VortexOpenOptions::file(opened).open().await? });

    let segment_map = vxf.file_layout().segment_map();
    let segment_origin = Vec::with_capacity(segment_map.len());

    vxf.file_layout().root_layout();

    println!("{}", full.tree_display());

    Ok(())
}
