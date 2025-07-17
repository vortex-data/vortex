// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use vortex::iter::ArrayIteratorExt;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;

pub fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let full = VortexOpenOptions::file()
        .open_blocking(file)?
        .scan()?
        .into_array_iter()?
        .read_all()?;

    println!("{}", full.display_tree());

    Ok(())
}
