// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::Parser;
use vortex_compat::validate::FixtureSource;
use vortex_compat::validate::discover_versions;
use vortex_compat::validate::validate_all;
use vortex_error::VortexResult;

#[derive(Parser)]
#[command(name = "validate", about = "Validate Vortex backward-compat fixtures")]
struct Cli {
    /// HTTPS base URL for the fixture bucket.
    /// e.g. <https://vortex-compat-fixtures.s3.amazonaws.com>
    #[arg(long)]
    fixtures_url: Option<String>,

    /// Local directory containing fixture versions (for development).
    #[arg(long)]
    fixtures_dir: Option<PathBuf>,

    /// Explicit list of versions to test (comma-separated).
    /// If omitted, discovers versions from versions.json or directory listing.
    #[arg(long, value_delimiter = ',')]
    versions: Option<Vec<String>>,
}

fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    let source = match (&cli.fixtures_url, &cli.fixtures_dir) {
        (Some(url), None) => FixtureSource::Url(url.clone()),
        (None, Some(dir)) => FixtureSource::Dir(dir.clone()),
        _ => {
            eprintln!("error: specify exactly one of --fixtures-url or --fixtures-dir");
            std::process::exit(1);
        }
    };

    let versions = match cli.versions {
        Some(v) => v,
        None => {
            eprintln!("discovering versions...");
            discover_versions(&source)?
        }
    };

    eprintln!(
        "testing {} version(s): {}",
        versions.len(),
        versions.join(", ")
    );

    let results = validate_all(&source, &versions)?;

    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut total_skipped = 0;

    for r in &results {
        total_passed += r.passed;
        total_failed += r.failed.len();
        total_skipped += r.skipped;
        if r.failed.is_empty() {
            eprintln!(
                "  v{}: {} passed, {} skipped",
                r.version, r.passed, r.skipped
            );
        } else {
            eprintln!(
                "  v{}: {} passed, {} FAILED, {} skipped",
                r.version,
                r.passed,
                r.failed.len(),
                r.skipped
            );
            for (name, err) in &r.failed {
                eprintln!("    FAIL {name}: {err}");
            }
        }
    }

    eprintln!("\nresult: {total_passed} passed, {total_failed} failed, {total_skipped} skipped");

    if total_failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}
