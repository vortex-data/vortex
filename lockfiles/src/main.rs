#![allow(clippy::expect_used)]

use std::fs::File;
use std::io::Write;
use std::process::Command;

use cargo_metadata::MetadataCommand;
use cargo_metadata::Package;
use cargo_metadata::PackageName;
use cargo_metadata::camino::Utf8PathBuf;
use indicatif::ParallelProgressIterator;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;

pub fn main() {
    let metadata = MetadataCommand::new()
        .no_deps()
        .exec()
        .expect("cargo metadata");

    let published: Vec<_> = metadata
        .workspace_packages()
        .into_iter()
        .filter(|v| is_published(v))
        // Only keep crates that publish Rust libs
        .filter(|p| p.targets.iter().any(|target| target.is_lib()))
        .collect();

    // Skip binary packages

    let progress = ProgressBar::new(published.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .expect("valid template")
            .progress_chars("=>-"),
    );

    published
        .par_iter()
        .progress_with(progress)
        .for_each(|pkg| {
            let job = LockfileJob {
                name: pkg.name.clone(),
                manifest_path: pkg.manifest_path.clone(),
            };

            job.execute().expect("lockfile");
        });
}

struct LockfileJob {
    name: PackageName,
    manifest_path: Utf8PathBuf,
}

impl LockfileJob {
    fn execute(self) -> std::io::Result<()> {
        let LockfileJob {
            name,
            manifest_path,
        } = self;

        let lockfile_path = manifest_path.with_file_name("public-api.lock");

        let mut cmd = Command::new("cargo");
        let output = cmd
            .arg("+nightly")
            .arg("public-api")
            .arg("--manifest-path")
            .arg(manifest_path)
            .args(["--omit", "blanket-impls,auto-trait-impls"])
            .output()?;

        if !output.status.success() {
            eprintln!(
                "FAILED: {name}:\n===============\n\n{}\n\n===============\n\n",
                String::from_utf8_lossy(&output.stdout)
            );

            return Err(std::io::Error::other("failed to execute"));
        }

        // Write lockfile contents
        File::create(&lockfile_path)?.write_all(&output.stdout)?;

        Ok(())
    }
}

fn is_published(pkg: &Package) -> bool {
    pkg.publish.as_ref().map(|v| !v.is_empty()).unwrap_or(true)
}
