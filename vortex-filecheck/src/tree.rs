// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checking every Vortex file under a directory that a check file applies to.

use std::fmt::Write as _;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::ensure;
use vortex::session::VortexSession;

use crate::check::CheckFile;
use crate::check::Glob;
use crate::check::SectionOutcome;
use crate::check::format_report;
use crate::units::DumpOptions;
use crate::units::check_vortex_file;

/// The outcome of checking one Vortex file.
#[derive(Debug)]
pub struct FileOutcome {
    /// The checked file.
    pub path: PathBuf,
    /// The outcome of each section of the check file.
    pub sections: Vec<SectionOutcome>,
}

impl FileOutcome {
    /// Whether every section passed.
    pub fn passed(&self) -> bool {
        self.sections.iter().all(SectionOutcome::passed)
    }

    /// Whether a section failed with [`crate::Severity::Error`], so the run fails.
    pub fn has_errors(&self) -> bool {
        self.sections.iter().any(SectionOutcome::is_error)
    }
}

/// Check every Vortex file under `root` whose path relative to `root` matches one of the
/// check file's `files:` regexes.
///
/// # Errors
///
/// Returns an error if the check file has no `files:` line, if its regexes match no Vortex
/// file, or if a file cannot be read. Failing checks are reported in the outcomes, not as an
/// error.
pub async fn check_tree(
    session: &VortexSession,
    root: &Path,
    check: &CheckFile,
    options: DumpOptions,
    excludes: &[Glob],
) -> anyhow::Result<Vec<FileOutcome>> {
    ensure!(
        !check.files.is_empty(),
        "the check file needs a `files: <regex>` line naming the Vortex files it checks"
    );
    let paths = matching_files(root, check)?;
    ensure!(
        !paths.is_empty(),
        "the `files:` regexes match no Vortex file under {}",
        root.display()
    );

    let mut outcomes = Vec::with_capacity(paths.len());
    for path in paths {
        let sections = check_vortex_file(session, &path, check, options, excludes).await?;
        outcomes.push(FileOutcome { path, sections });
    }
    Ok(outcomes)
}

/// The `.vortex` files under `root`, sorted, whose paths relative to `root` (with `/`
/// separators on every platform) match one of the check file's `files:` regexes.
///
/// # Errors
///
/// Returns an error if a directory cannot be read.
pub fn matching_files(root: &Path, check: &CheckFile) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_vortex_files(root, &mut paths)?;
    paths.retain(|path| check.applies_to(&relative_path(root, path)));
    paths.sort();
    Ok(paths)
}

fn collect_vortex_files(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        // `file_type` does not follow symlinks, so a symlink cycle cannot recurse forever.
        if entry.file_type()?.is_dir() {
            collect_vortex_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "vortex") {
            out.push(path);
        }
    }
    Ok(())
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Describe the outcomes of [`check_tree`]: [`format_report`] for each file, under a heading
/// naming the file.
pub fn format_tree_report(outcomes: &[FileOutcome], explain: bool) -> String {
    let mut out = String::new();
    for outcome in outcomes {
        let status = if outcome.passed() {
            "PASS"
        } else if outcome.has_errors() {
            "FAIL"
        } else {
            "WARN"
        };
        // Writing to a `String` cannot fail.
        writeln!(out, "{status} {}", outcome.path.display()).unwrap_or_default();
        for line in format_report(&outcome.sections, explain).lines() {
            writeln!(out, "  {line}").unwrap_or_default();
        }
    }
    out
}
