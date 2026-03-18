// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CLI for managing a benchmark dataset repository.
//!
//! ```text
//! bench-data init my-dataset                      # scaffold a new dataset
//! bench-data manifest my-dataset/                 # generate manifest from files
//! bench-data validate my-dataset/                 # check dataset is correct
//! bench-data push my-dataset/ --remote <url>      # upload to remote
//! bench-data pull --remote <url> --local <dir>    # fetch catalog + manifests
//! bench-data checkout tpch-sf100 --remote <url>   # download data files
//! bench-data list --local <dir>                   # list datasets
//! bench-data describe tpch-sf100 --local <dir>    # show dataset details
//! bench-data delete tpch-sf100 --remote <url>     # remove dataset
//! bench-data gc --remote <url>                    # clean orphaned data
//! bench-data verify tpch-sf100 --remote <url>     # check integrity
//! ```

use std::path::PathBuf;

use clap::Parser;
use clap::Subcommand;
use humansize::BINARY;
use humansize::format_size;
use tracing::error;
use vortex_bench::datagen::catalog::Catalog;
use vortex_bench::datagen::dataset::DatasetDescriptor;
use vortex_bench::datagen::local;
use vortex_bench::datagen::manifest::Manifest;
use vortex_bench::datagen::remote;
use vortex_bench::setup_logging_and_tracing;

#[derive(Parser)]
#[command(name = "bench-data")]
#[command(about = "Manage a benchmark dataset repository")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a new dataset directory with template dataset.yaml.
    Init {
        /// Dataset name (lowercase, hyphens, numbers).
        name: String,
        /// Parent directory to create the dataset in (default: current directory).
        #[arg(short, long, default_value = ".")]
        dir: PathBuf,
    },
    /// Generate manifest.json by scanning data files and computing hashes.
    Manifest {
        /// Path to the dataset directory.
        path: PathBuf,
    },
    /// Validate a dataset directory before pushing.
    Validate {
        /// Path to the dataset directory.
        path: PathBuf,
    },
    /// Upload a dataset to the remote repository.
    Push {
        /// Path to the local dataset directory.
        path: PathBuf,
        /// Remote URL (s3://bucket/prefix, gs://bucket/prefix, or local path).
        #[arg(short, long)]
        remote: String,
        /// Overwrite existing dataset without prompting.
        #[arg(long, default_value = "false")]
        force: bool,
    },
    /// Fetch catalog and manifests from remote (no data files).
    Pull {
        /// Remote URL.
        #[arg(short, long)]
        remote: String,
        /// Local mirror directory.
        #[arg(short, long, default_value = "~/.cache/vortex-bench-data")]
        local: PathBuf,
    },
    /// Download data files for a dataset.
    Checkout {
        /// Dataset name.
        name: String,
        /// Remote URL.
        #[arg(short, long)]
        remote: String,
        /// Local mirror directory.
        #[arg(short, long, default_value = "~/.cache/vortex-bench-data")]
        local: PathBuf,
    },
    /// List all datasets in the catalog.
    List {
        /// Local mirror directory (after pull) or remote URL.
        #[arg(short, long, default_value = "~/.cache/vortex-bench-data")]
        local: PathBuf,
    },
    /// Show details for a dataset.
    Describe {
        /// Dataset name.
        name: String,
        /// Local mirror directory.
        #[arg(short, long, default_value = "~/.cache/vortex-bench-data")]
        local: PathBuf,
    },
    /// Remove a dataset from the catalog.
    Delete {
        /// Dataset name.
        name: String,
        /// Remote URL.
        #[arg(short, long)]
        remote: String,
        /// Also delete data files from remote.
        #[arg(long, default_value = "false")]
        purge: bool,
        /// Skip confirmation prompt (required for non-interactive use).
        #[arg(long, default_value = "false")]
        force: bool,
    },
    /// Clean orphaned data directories not referenced by the catalog.
    Gc {
        /// Remote URL.
        #[arg(short, long)]
        remote: String,
    },
    /// Verify integrity of a dataset (check all hashes).
    Verify {
        /// Dataset name.
        name: String,
        /// Remote URL.
        #[arg(short, long)]
        remote: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    setup_logging_and_tracing(cli.verbose, false)?;

    match cli.command {
        Command::Init { name, dir } => {
            let dataset_dir = dir.join(&name);
            local::init(&dataset_dir, &name)?;
            println!("Initialized dataset at {}", dataset_dir.display());
            println!("  1. Edit {}/dataset.yaml", dataset_dir.display());
            println!(
                "  2. Add files to {}/data/{{format}}/{{table}}/",
                dataset_dir.display()
            );
            println!(
                "  3. Run: bench-data push {} --remote <url>",
                dataset_dir.display()
            );
        }

        Command::Manifest { path } => {
            let manifest = local::manifest(&path)?;
            println!(
                "Generated manifest: {} files, {}",
                manifest.total_files(),
                format_size(manifest.total_size_bytes(), BINARY)
            );
        }

        Command::Validate { path } => {
            let problems = local::validate(&path)?;
            if problems.is_empty() {
                println!("Validation passed");
            } else {
                println!("Validation failed:");
                for p in &problems {
                    println!("  - {p}");
                }
                std::process::exit(1);
            }
        }

        Command::Push {
            path,
            remote: remote_url,
            force,
        } => {
            let (store, base) = remote::resolve_store(&remote_url)?;

            // If not forced, check for existing and prompt.
            let should_force = if !force {
                if let Some(existing) = remote::check_existing(store.as_ref(), &base, &{
                    // Read the dataset name from descriptor to check.
                    let desc = DatasetDescriptor::from_file(path.join("dataset.yaml"))?;
                    desc.name
                })
                .await?
                {
                    eprintln!(
                        "Dataset '{}' already exists at '{}'.",
                        existing.name, existing.path
                    );
                    eprint!("Replace it? [y/N] ");
                    let mut answer = String::new();
                    std::io::stdin().read_line(&mut answer)?;
                    if !answer.trim().eq_ignore_ascii_case("y") {
                        println!("Aborted.");
                        return Ok(());
                    }
                    true
                } else {
                    false
                }
            } else {
                true
            };

            remote::push(store.as_ref(), &base, &path, force || should_force).await?;
            println!("Push complete");
        }

        Command::Pull {
            remote: remote_url,
            local: local_dir,
        } => {
            let local_dir = expand_tilde(&local_dir);
            let (store, base) = remote::resolve_store(&remote_url)?;
            remote::pull(store.as_ref(), &base, &local_dir).await?;
            println!("Pull complete");
        }

        Command::Checkout {
            name,
            remote: remote_url,
            local: local_dir,
        } => {
            let local_dir = expand_tilde(&local_dir);
            let (store, base) = remote::resolve_store(&remote_url)?;
            remote::checkout(store.as_ref(), &base, &local_dir, &name).await?;
            println!("Checkout complete");
        }

        Command::List { local: local_dir } => {
            let local_dir = expand_tilde(&local_dir);
            let catalog_path = local_dir.join("catalog.json");
            if !catalog_path.exists() {
                println!("No local catalog found. Run `bench-data pull` first.");
                return Ok(());
            }
            let bytes = std::fs::read(&catalog_path)?;
            let catalog = Catalog::from_json(&bytes)?;
            if catalog.datasets.is_empty() {
                println!("No datasets in catalog.");
            } else {
                println!("{:<30} PATH", "NAME");
                for entry in &catalog.datasets {
                    println!("{:<30} {}", entry.name, entry.path);
                }
            }
        }

        Command::Describe {
            name,
            local: local_dir,
        } => {
            let local_dir = expand_tilde(&local_dir);
            let catalog_path = local_dir.join("catalog.json");
            let bytes = std::fs::read(&catalog_path)?;
            let catalog = Catalog::from_json(&bytes)?;

            let entry = catalog
                .find(&name)
                .ok_or_else(|| anyhow::anyhow!("dataset '{}' not found", name))?;

            let dataset_dir = local_dir.join(&entry.path);

            // Show descriptor if available.
            let descriptor_path = dataset_dir.join("dataset.yaml");
            if descriptor_path.exists() {
                let desc = DatasetDescriptor::from_file(&descriptor_path)?;
                println!("Name:        {}", desc.name);
                println!("Description: {}", desc.description);
                println!("Author:      {}", desc.author);
                if !desc.tags.is_empty() {
                    println!("Tags:        {}", desc.tags.join(", "));
                }
                if let Some(source) = &desc.source {
                    println!("Source:");
                    println!("  Kind:        {}", source.kind);
                    println!("  Description: {}", source.description);
                    if let Some(cmd) = &source.command {
                        println!("  Command:     {cmd}");
                    }
                    if let Some(parent) = &source.parent {
                        println!("  Parent:      {parent}");
                    }
                    if let Some(url) = &source.url {
                        println!("  URL:         {url}");
                    }
                }
            }

            // Show manifest if available.
            let manifest_path = dataset_dir.join("manifest.json");
            if manifest_path.exists() {
                let manifest_bytes = std::fs::read(&manifest_path)?;
                let manifest = Manifest::from_json(&manifest_bytes)?;
                println!();
                println!(
                    "Files: {} total, {}",
                    manifest.total_files(),
                    format_size(manifest.total_size_bytes(), BINARY)
                );
                for (format, tables) in &manifest.formats {
                    for (table, te) in &tables.tables {
                        for file in &te.files {
                            println!(
                                "  [{format}/{table}] {} ({})",
                                file.path,
                                format_size(file.size_bytes, BINARY)
                            );
                        }
                    }
                }
            }
        }

        Command::Delete {
            name,
            remote: remote_url,
            purge,
            force,
        } => {
            let (store, base) = remote::resolve_store(&remote_url)?;

            if !force {
                if purge {
                    eprintln!(
                        "This will permanently delete '{name}' from the catalog \
                         AND remove all data files from remote storage."
                    );
                } else {
                    eprintln!(
                        "This will remove '{name}' from the catalog. \
                         Data files will remain in remote storage until `gc`."
                    );
                }
                eprint!("Continue? [y/N] ");
                let mut answer = String::new();
                std::io::stdin().read_line(&mut answer)?;
                if !answer.trim().eq_ignore_ascii_case("y") {
                    println!("Aborted.");
                    return Ok(());
                }
            }

            remote::delete(store.as_ref(), &base, &name, purge).await?;
            println!("Deleted '{name}' from catalog");
        }

        Command::Gc { remote: remote_url } => {
            let (store, base) = remote::resolve_store(&remote_url)?;
            let removed = remote::gc(store.as_ref(), &base).await?;
            if removed.is_empty() {
                println!("Nothing to clean up");
            } else {
                for path in &removed {
                    println!("Removed: {path}");
                }
            }
        }

        Command::Verify {
            name,
            remote: remote_url,
        } => {
            let (store, base) = remote::resolve_store(&remote_url)?;
            let problems = remote::verify(store.as_ref(), &base, &name).await?;
            if problems.is_empty() {
                println!("Verification passed");
            } else {
                println!("Verification failed:");
                for p in &problems {
                    error!("  - {p}");
                }
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

/// Expand ~ to home directory.
fn expand_tilde(path: &std::path::Path) -> PathBuf {
    if let (Ok(stripped), Some(home)) = (path.strip_prefix("~"), dirs_home()) {
        return home.join(stripped);
    }
    path.to_path_buf()
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
