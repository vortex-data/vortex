// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The code identity every task key and measurement is indexed by.

use std::path::Path;
use std::process::Command;

use anyhow::Context;
use anyhow::bail;
use serde::Deserialize;
use serde::Serialize;

/// The git commit this binary was built from, or `unknown`.
pub const BUILD_GIT_COMMIT: &str = env!("VX_LAB_GIT_COMMIT");

/// The Vortex workspace version this binary was built from.
pub const BUILD_VX_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Identifies the code that produced a result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeIdentity {
    /// The Vortex workspace version.
    pub vx_version: String,
    /// The full git commit hash.
    pub git_commit: String,
    /// Whether the working tree had uncommitted changes to tracked files.
    ///
    /// Dirty results get different task keys, so they never mix with results from the commit.
    pub dirty: bool,
}

impl CodeIdentity {
    /// The identity of this build, checked against the checkout it is run from.
    ///
    /// Fails if the binary was built from a different commit than the checkout's `HEAD`, or if
    /// tracked files are modified and `allow_dirty` is false.
    pub fn of_this_build(allow_dirty: bool) -> anyhow::Result<Self> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");

        if BUILD_GIT_COMMIT == "unknown" {
            if !allow_dirty {
                bail!(
                    "this binary was built without a git commit; build it from a git checkout, \
                     or pass --allow-dirty to plan anyway"
                );
            }
            return Ok(Self {
                vx_version: BUILD_VX_VERSION.to_string(),
                git_commit: BUILD_GIT_COMMIT.to_string(),
                dirty: true,
            });
        }

        let head = git(&repo_root, &["rev-parse", "HEAD"])?;
        if head != BUILD_GIT_COMMIT {
            bail!(
                "this binary was built from {BUILD_GIT_COMMIT} but the checkout is at {head}; \
                 rebuild before planning"
            );
        }

        let dirty = !git(&repo_root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty();
        if dirty && !allow_dirty {
            bail!(
                "tracked files have uncommitted changes; commit them, or pass --allow-dirty to \
                 plan with keys marked dirty"
            );
        }

        Ok(Self {
            vx_version: BUILD_VX_VERSION.to_string(),
            git_commit: BUILD_GIT_COMMIT.to_string(),
            dirty,
        })
    }
}

fn git(repo_root: &Path, args: &[&str]) -> anyhow::Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}
