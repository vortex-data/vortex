// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::scan::rayon::ParallelArrayIteratorExt;

pub fn exec_tree(file: impl AsRef<Path>) -> VortexResult<()> {
    let full = VortexOpenOptions::file()
        .open_blocking(file)?
        .scan()?
        .into_par_iter()?
        .read_all()?;

    println!("{}", full.display_tree());

    Ok(())
}
