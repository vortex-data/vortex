// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::stream::ArrayStreamExt;

pub async fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let full = VortexOpenOptions::new()
        .open(file.as_ref())
        .await?
        .scan()?
        .into_tokio_array_stream()?
        .read_all()
        .await?;

    println!("{}", full.display_tree());

    Ok(())
}
