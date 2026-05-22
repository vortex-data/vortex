// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod constants;
pub mod file;
pub mod logging;

use std::process::Command;
use std::sync::LazyLock;

/// Re-export of `vortex::utils::aliases` so downstream benchmark crates can
/// reach the standard `HashMap`/`HashSet` aliases without pulling in `vortex`
/// directly.
pub use vortex::utils::aliases;

pub static GIT_COMMIT_ID: LazyLock<String> = LazyLock::new(|| {
    String::from_utf8(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_string()
});
