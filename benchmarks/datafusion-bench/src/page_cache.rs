// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `--opt evict-page-cache=on`: before every run, the local data's pages leave the OS's page
//! cache, so each run reads the files from the drive as a fresh process's first run does, but in
//! one process. Its later runs then compare with a cache's warm runs: the process, its threads and
//! the footers DataFusion keeps are warm on both sides.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use url::Url;

/// The local data's files, whose pages [`Self::evict`] drops.
pub struct PageCacheEviction {
    files: Vec<PathBuf>,
}

impl PageCacheEviction {
    /// Every file under the directory `data_url` names.
    pub fn of_local_data(data_url: &Url) -> anyhow::Result<Self> {
        let dir = data_url
            .to_file_path()
            .map_err(|()| anyhow::anyhow!("evict-page-cache needs local data, not {data_url}"))?;
        let mut files = Vec::new();
        Self::collect(&dir, &mut files)?;
        anyhow::ensure!(!files.is_empty(), "no files under {}", dir.display());
        Ok(Self { files })
    }

    fn collect(dir: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                Self::collect(&path, files)?;
            } else {
                files.push(path);
            }
        }
        Ok(())
    }

    /// Drops the files' pages, which are clean: the data was written before the process began.
    pub fn evict(&self) -> anyhow::Result<()> {
        self.files
            .iter()
            .try_for_each(|path| Self::evict_file(path))
    }

    #[cfg(target_os = "linux")]
    fn evict_file(path: &Path) -> anyhow::Result<()> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        rustix::fs::fadvise(&file, 0, None, rustix::fs::Advice::DontNeed)
            .with_context(|| format!("evicting {}", path.display()))
    }

    #[cfg(not(target_os = "linux"))]
    fn evict_file(_path: &Path) -> anyhow::Result<()> {
        anyhow::bail!("evict-page-cache needs Linux's posix_fadvise")
    }
}
