// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Recompress every field of a Vortex file with the BtrBlocks compressor and
// print the resulting (most-compressed) array tree, plus per-top-level-field
// before/after sizes.
#![allow(deprecated)]

use std::path::PathBuf;

use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::compressor::BtrBlocksCompressor;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

#[tokio::main]
async fn main() -> VortexResult<()> {
    let file = PathBuf::from(std::env::args().nth(1).expect("usage: recompress <file>"));
    let session = VortexSession::default().with_tokio().allow_unknown();

    let array = session
        .open_options()
        .open_path(&file)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    let before = array.nbytes();
    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&array, &mut session.create_execution_ctx())?;
    let after = compressed.nbytes();

    println!(
        "TOTAL: {before} -> {after} bytes  ({:.2}x, {:.1}% of original)\n",
        before as f64 / after as f64,
        100.0 * after as f64 / before as f64
    );
    println!("=== most-compressed array tree (per-field nbytes shown) ===");
    println!("{}", compressed.display_tree());
    Ok(())
}
