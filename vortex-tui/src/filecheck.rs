// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vx filecheck`: check the array encodings of a Vortex file with FileCheck-style directives.
//!
//! See [`vortex_filecheck`] for the dump and check-file formats. `vx filecheck` checks the file
//! it is given, so it ignores a check file's `files:` lines.

use std::path::PathBuf;

use anyhow::bail;
use vortex::session::VortexSession;
use vortex_filecheck::DumpOptions;
use vortex_filecheck::Glob;
use vortex_filecheck::SectionOutcome;
use vortex_filecheck::format_report;
use vortex_filecheck::load_units;
use vortex_filecheck::read_check_file;
use vortex_filecheck::run_checks;

/// Command-line arguments for the filecheck command.
#[derive(Debug, clap::Parser)]
pub struct FilecheckArgs {
    /// Path to the Vortex file.
    pub file: PathBuf,
    /// File of check directives. Without it, the text being checked is printed instead.
    pub checks: Option<PathBuf>,
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
}

/// Run `vx filecheck`.
///
/// # Errors
///
/// Returns an error if the file cannot be read, the check file is malformed, or a check fails.
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
    let units = load_units(session, &args.file, options, &excludes).await?;

    let Some(checks_path) = args.checks else {
        for unit in &units {
            print!("{}", unit.text);
        }
        return Ok(());
    };

    let check = read_check_file(&checks_path)?;
    let mut outcomes = run_checks(&units, &check.sections)?;
    if args.warn_only {
        outcomes
            .iter_mut()
            .for_each(SectionOutcome::downgrade_to_warning);
    }
    print!("{}", format_report(&outcomes, args.explain));

    let failed = outcomes.iter().filter(|outcome| outcome.is_error()).count();
    if failed > 0 {
        bail!("{failed} of {} check sections failed", outcomes.len());
    }
    Ok(())
}
