// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use vortex_compat::generate::generate;
use vortex_compat::publish::publish;
use vortex_compat::store::DEFAULT_STORE;
use vortex_compat::store::FixtureStore;
use vortex_compat::store::parse_store;
use vortex_compat::validate::run_check;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

#[derive(Parser)]
#[command(
    name = "vortex-compat",
    about = "Vortex backward-compatibility fixture management"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate fixture files into a local directory.
    ///
    /// Builds all fixtures in memory first, then writes to disk.
    /// If any fixture fails to build, nothing is written.
    Generate {
        /// Version tag (e.g. "0.63.0").
        #[arg(long)]
        version: String,

        /// Local directory to write fixtures into.
        #[arg(long)]
        output: PathBuf,
    },

    /// Generate fixtures and publish them to a store.
    ///
    /// Stages: (1) generate all fixtures locally, (2) merge manifest with
    /// previous version, (3) upload all fixture files in parallel,
    /// (4) write manifest, (5) update versions.json.
    /// Nothing is uploaded until all fixtures are successfully generated.
    Publish {
        /// Version tag (e.g. "0.63.0").
        #[arg(long)]
        version: String,

        /// Fixture store: local path or s3://bucket.
        #[arg(long, default_value = DEFAULT_STORE)]
        store: String,

        /// Local directory for intermediate fixture files (default: temp dir).
        #[arg(long)]
        tmp_dir: Option<PathBuf>,

        /// Generate and merge manifest but don't upload to the store.
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate fixtures in a store against the current reader.
    Check {
        /// Fixture store: local path or s3://bucket.
        #[arg(long, default_value = DEFAULT_STORE)]
        store: String,

        /// Comma-separated versions to validate (default: all).
        #[arg(long, value_delimiter = ',')]
        versions: Option<Vec<String>>,
    },

    /// List versions and fixtures in a store.
    List {
        /// Fixture store: local path or s3://bucket.
        #[arg(long, default_value = DEFAULT_STORE)]
        store: String,

        /// Show detailed manifest for a specific version.
        #[arg(long)]
        version: Option<String>,
    },

    /// Validate that manifests are additive-only across all versions.
    ///
    /// Checks that each version's manifest contains all fixtures from the
    /// previous version (and possibly more). Does not read fixture files,
    /// only manifests.
    ValidateManifest {
        /// Fixture store: local path or s3://bucket.
        #[arg(long, default_value = DEFAULT_STORE)]
        store: String,
    },
}

fn main() -> VortexResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { version, output } => generate(&output, &version),
        Commands::Publish {
            version,
            store,
            tmp_dir,
            dry_run,
        } => {
            let store = parse_store(&store)?;
            publish(store.as_ref(), &version, tmp_dir.as_deref(), dry_run)
        }
        Commands::Check { store, versions } => {
            let store = parse_store(&store)?;
            run_check(store.as_ref(), versions)
        }
        Commands::List { store, version } => {
            let store = parse_store(&store)?;
            run_list(store.as_ref(), version)
        }
        Commands::ValidateManifest { store } => {
            let store = parse_store(&store)?;
            run_validate_manifest(store.as_ref())
        }
    }
}

fn run_list(store: &dyn FixtureStore, version: Option<String>) -> VortexResult<()> {
    if let Some(ver) = version {
        let manifest = store.fetch_manifest(&ver)?;
        eprintln!(
            "v{} (generated {}):",
            manifest.version, manifest.generated_at
        );
        for entry in &manifest.fixtures {
            eprintln!("  {:<30} (since {})", entry.name, entry.since);
        }
    } else {
        let versions = store.list_versions()?;
        eprintln!("Versions ({}):", store.display_name());
        if versions.is_empty() {
            eprintln!("  (none)");
        } else {
            for v in &versions {
                eprintln!("  {v}");
            }
        }
    }
    Ok(())
}

/// Validate that manifests are additive-only across all versions.
fn run_validate_manifest(store: &dyn FixtureStore) -> VortexResult<()> {
    let versions = store.list_versions()?;
    if versions.is_empty() {
        eprintln!("no versions found in {}", store.display_name());
        return Ok(());
    }

    eprintln!(
        "validating manifests for {} version(s) in {}...",
        versions.len(),
        store.display_name()
    );

    let mut prev_fixtures: Option<(String, vortex_utils::aliases::hash_set::HashSet<String>)> =
        None;
    let mut errors = Vec::new();

    for version in &versions {
        let manifest = store.fetch_manifest(version)?;
        let fixture_names: vortex_utils::aliases::hash_set::HashSet<String> =
            manifest.fixtures.iter().map(|e| e.name.clone()).collect();

        if let Some((prev_version, ref prev_names)) = prev_fixtures {
            let missing: Vec<&String> = prev_names
                .iter()
                .filter(|name| !fixture_names.contains(name.as_str()))
                .collect();

            if missing.is_empty() {
                let new_count = fixture_names.len() - prev_names.len();
                if new_count > 0 {
                    eprintln!(
                        "  v{prev_version} -> v{version}: ok ({} fixtures, +{new_count} new)",
                        fixture_names.len()
                    );
                } else {
                    eprintln!(
                        "  v{prev_version} -> v{version}: ok ({} fixtures)",
                        fixture_names.len()
                    );
                }
            } else {
                let missing_list: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                eprintln!(
                    "  v{prev_version} -> v{version}: FAIL — missing: {}",
                    missing_list.join(", ")
                );
                errors.push(format!(
                    "v{version} is missing fixtures from v{prev_version}: {}",
                    missing_list.join(", ")
                ));
            }
        } else {
            eprintln!(
                "  v{version}: {} fixtures (first version)",
                fixture_names.len()
            );
        }

        prev_fixtures = Some((version.clone(), fixture_names));
    }

    if errors.is_empty() {
        eprintln!("\nall manifests are additive-only.");
        Ok(())
    } else {
        eprintln!("\n{} error(s) found.", errors.len());
        vortex_bail!("manifest validation failed:\n{}", errors.join("\n"));
    }
}
