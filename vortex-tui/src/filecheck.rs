// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vx filecheck`: check the array encodings of Vortex files with FileCheck-style directives.
//!
//! See [`vortex_filecheck`] for the dump and check-file formats. There are three modes:
//!
//! - `vx filecheck <file>` prints the dump of a file, to help write checks.
//! - `vx filecheck <file> <check>` checks one file. It ignores the check file's `files:` lines.
//! - `vx filecheck --root <dir> <check>...` checks every Vortex file under `dir` that a check
//!   file's `files:` regexes match, and skips check files that match none, so one list of check
//!   files can run against any dataset.

use std::fmt::Write as _;
use std::path::Path;
use std::path::PathBuf;

use anyhow::bail;
use vortex::session::VortexSession;
use vortex_filecheck::DumpOptions;
use vortex_filecheck::FileOutcome;
use vortex_filecheck::Glob;
use vortex_filecheck::SectionOutcome;
use vortex_filecheck::format_report;
use vortex_filecheck::load_units;
use vortex_filecheck::matching_files;
use vortex_filecheck::read_check_file;
use vortex_filecheck::run_checks;

/// Command-line arguments for the filecheck command.
#[derive(Debug, clap::Parser)]
pub struct FilecheckArgs {
    /// Without `--root`: the Vortex file, then optionally a check file. Without a check file,
    /// the text being checked is printed instead. With `--root`: one or more check files.
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
    /// Check every Vortex file under this directory that a check file's `files:` regexes match,
    /// with paths relative to it.
    #[arg(long)]
    pub root: Option<PathBuf>,
    /// Skip units whose path matches this glob. Can be repeated.
    #[arg(long)]
    pub exclude: Vec<String>,
    /// Include array metadata in the dump.
    #[arg(long)]
    pub metadata: bool,
    /// Include array sizes in the dump.
    #[arg(long)]
    pub nbytes: bool,
    /// Include array buffers in the dump.
    #[arg(long)]
    pub buffers: bool,
    /// Include array statistics in the dump.
    #[arg(long)]
    pub stats: bool,
    /// Print the full match trace for each failure instead of the lines around it.
    #[arg(long)]
    pub explain: bool,
    /// Report every failing section as a warning and exit successfully.
    #[arg(long)]
    pub warn_only: bool,
    /// How to print the results of checks.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
}

impl FilecheckArgs {
    /// The path that must exist: the root directory, or the Vortex file.
    pub fn input_path(&self) -> Option<&PathBuf> {
        self.root.as_ref().or_else(|| self.paths.first())
    }
}

/// How `vx filecheck` prints the results of checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Plain text, one section per line.
    Text,
    /// A Markdown summary table with the failures in a collapsed block, for PR comments. Prints
    /// nothing when no check file matched any file.
    Markdown,
}

/// The checks of one check file against the files it matched.
struct CheckRun {
    check: PathBuf,
    files: Vec<FileOutcome>,
}

/// Run `vx filecheck`.
///
/// # Errors
///
/// Returns an error if a file cannot be read, a check file is malformed, or a section with
/// `severity: error` fails without `--warn-only`.
pub async fn exec_filecheck(session: &VortexSession, args: FilecheckArgs) -> anyhow::Result<()> {
    let options = DumpOptions {
        metadata: args.metadata,
        nbytes: args.nbytes,
        buffers: args.buffers,
        stats: args.stats,
    };
    let excludes = args
        .exclude
        .iter()
        .map(|pattern| Glob::new(pattern))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut runs = match (&args.root, args.paths.as_slice()) {
        (Some(root), checks) => check_root(session, root, checks, options, &excludes).await?,
        (None, [file]) => {
            for unit in load_units(session, file, options, &excludes).await? {
                print!("{}", unit.text);
            }
            return Ok(());
        }
        (None, [file, check]) => {
            let units = load_units(session, file, options, &excludes).await?;
            let sections = run_checks(&units, &read_check_file(check)?.sections)?;
            vec![CheckRun {
                check: check.clone(),
                files: vec![FileOutcome {
                    path: file.clone(),
                    sections,
                }],
            }]
        }
        (None, _) => bail!("without --root, pass a Vortex file and at most one check file"),
    };

    if args.warn_only {
        for run in &mut runs {
            for file in &mut run.files {
                file.sections
                    .iter_mut()
                    .for_each(SectionOutcome::downgrade_to_warning);
            }
        }
    }

    match args.format {
        Format::Text => print!("{}", text_report(&runs, args.root.is_some(), args.explain)),
        Format::Markdown => print!("{}", markdown_report(&runs, args.root.as_deref())),
    }

    let failed = runs
        .iter()
        .flat_map(|run| &run.files)
        .filter(|file| file.has_errors())
        .count();
    if failed > 0 {
        bail!("encoding checks failed for {failed} file(s)");
    }
    Ok(())
}

/// Check every Vortex file under `root` that each check file applies to. A check file that
/// applies to no file is kept with no outcomes, and reported as skipped.
async fn check_root(
    session: &VortexSession,
    root: &Path,
    checks: &[PathBuf],
    options: DumpOptions,
    excludes: &[Glob],
) -> anyhow::Result<Vec<CheckRun>> {
    let mut runs = Vec::with_capacity(checks.len());
    for check_path in checks {
        let check = read_check_file(check_path)?;
        let mut files = Vec::new();
        for path in matching_files(root, &check)? {
            let units = load_units(session, &path, options, excludes).await?;
            let sections = run_checks(&units, &check.sections)?;
            files.push(FileOutcome { path, sections });
        }
        runs.push(CheckRun {
            check: check_path.clone(),
            files,
        });
    }
    Ok(runs)
}

fn text_report(runs: &[CheckRun], headings: bool, explain: bool) -> String {
    let mut out = String::new();
    for run in runs {
        if run.files.is_empty() {
            // Writing to a `String` cannot fail.
            writeln!(out, "SKIP {}: no matching files", run.check.display()).unwrap_or_default();
        }
        for file in &run.files {
            if headings {
                writeln!(out, "== {} on {}", run.check.display(), file.path.display())
                    .unwrap_or_default();
            }
            out.push_str(&format_report(&file.sections, explain));
        }
    }
    out
}

fn markdown_report(runs: &[CheckRun], root: Option<&Path>) -> String {
    if runs.iter().all(|run| run.files.is_empty()) {
        return String::new();
    }
    let display = |path: &Path| {
        root.and_then(|root| path.strip_prefix(root).ok())
            .unwrap_or(path)
            .display()
            .to_string()
    };

    let mut out = String::new();
    let mut details = String::new();
    // Writing to a `String` cannot fail.
    writeln!(out, "### Encoding checks\n").unwrap_or_default();
    writeln!(out, "| Check | File | Result |").unwrap_or_default();
    writeln!(out, "| --- | --- | --- |").unwrap_or_default();
    for run in runs {
        for file in &run.files {
            let passed = file.sections.iter().filter(|s| s.passed()).count();
            let warned = file.sections.iter().filter(|s| s.is_warning()).count();
            let failed = file.sections.iter().filter(|s| s.is_error()).count();
            let icon = if failed > 0 {
                "❌"
            } else if warned > 0 {
                "⚠️"
            } else {
                "✅"
            };
            writeln!(
                out,
                "| `{}` | `{}` | {icon} {passed} passed, {warned} warned, {failed} failed |",
                run.check.display(),
                display(&file.path),
            )
            .unwrap_or_default();
            if !file.passed() {
                writeln!(
                    details,
                    "{} on {}:\n{}",
                    run.check.display(),
                    display(&file.path),
                    format_report(&file.sections, false)
                )
                .unwrap_or_default();
            }
        }
    }
    let skipped: Vec<String> = runs
        .iter()
        .filter(|run| run.files.is_empty())
        .map(|run| format!("`{}`", run.check.display()))
        .collect();
    if !skipped.is_empty() {
        writeln!(out, "\nNo matching files: {}", skipped.join(", ")).unwrap_or_default();
    }
    if !details.is_empty() {
        writeln!(
            out,
            "\n<details><summary>Warnings and failures</summary>\n\n```text\n{}```\n\n</details>",
            details
        )
        .unwrap_or_default();
    }
    out
}
