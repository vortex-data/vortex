// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;

/// Recursively collects every `.slt` file under `path`.
///
/// `.slt.no` include fragments are intentionally skipped; they are pulled in by
/// `include` directives rather than run as standalone files.
pub fn list_files(path: impl AsRef<Path>) -> anyhow::Result<Vec<PathBuf>> {
    list_files_with_extension(path, "slt")
}

/// Recursively collects every file under `path` whose extension is exactly `extension`.
pub fn list_files_with_extension(
    path: impl AsRef<Path>,
    extension: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut file_paths = vec![];

    list_files_impl(&mut file_paths, path, extension)?;

    Ok(file_paths)
}

fn list_files_impl(
    file_paths: &mut Vec<PathBuf>,
    path: impl AsRef<Path>,
    extension: &str,
) -> anyhow::Result<()> {
    let path = path.as_ref();

    let read_dir = std::fs::read_dir(path)?;
    for entry in read_dir {
        let entry = entry?;

        if entry.metadata()?.is_dir() {
            list_files_impl(file_paths, entry.path(), extension)?;
        } else {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == extension) {
                file_paths.push(entry.path());
            }
        }
    }

    Ok(())
}
