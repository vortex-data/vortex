// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-test scratch directory management.
//!
//! Each test writes its artifacts to a deterministic directory beneath
//! [`scratch_root`], exposed to SQL as `${WORK_DIR}`. [`work_dir_for`] derives a
//! test's directory from its (unique) name, [`reset_dir`] gives it a clean slate
//! before the run, and [`WorkDirGuard`] removes it afterwards — whether the test
//! passed, failed, or panicked.
//!
//! Output that references these paths is rewritten back to the `${WORK_DIR}`
//! token by [`crate::normalize`].

use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Root of the scratch directory used for test artifacts, inside this crate.
///
/// Each test uses a unique subdirectory beneath this root.
pub fn scratch_root() -> PathBuf {
    static SCRATCH_DIR: LazyLock<PathBuf> =
        LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scratch"));

    SCRATCH_DIR.clone()
}

/// The scratch directory `${WORK_DIR}` resolves to for a single test.
///
/// It is a deterministic (not random) path under [`scratch_root`], derived from
/// the test's unique name, so concurrent tests never collide.
pub fn work_dir_for(test_name: &str) -> PathBuf {
    scratch_root().join(test_name.replace([':', '/', '\\'], "_"))
}

/// Recreates `dir` empty, clearing anything left behind by a previous run.
pub fn reset_dir(dir: &Path) -> anyhow::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    Ok(())
}

/// Removes a test's scratch directory on drop — whether the test passed, failed,
/// or panicked. Cleanup errors are logged, not propagated.
pub struct WorkDirGuard(PathBuf);

impl WorkDirGuard {
    /// Creates a guard that removes `dir` (and the emptied scratch root) on drop.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self(dir.into())
    }
}

impl Drop for WorkDirGuard {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "warning: failed to clean scratch dir {}: {e}",
                self.0.display()
            );
        }
        // Best-effort removal of the scratch root once the last test empties it;
        // fails harmlessly while other tests still have directories there.
        std::fs::remove_dir(scratch_root()).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::scratch_root;
    use super::work_dir_for;

    #[test]
    fn work_dir_is_under_scratch_root_with_sanitized_name() {
        let dir = work_dir_for("slt::duckdb::duckdb/explain.slt");
        assert_eq!(dir.parent(), Some(scratch_root().as_path()));
        assert_eq!(dir.file_name().unwrap(), "slt__duckdb__duckdb_explain.slt");
    }
}
