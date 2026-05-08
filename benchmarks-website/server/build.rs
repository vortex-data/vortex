// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::process::Command;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-lib=dylib=rstrtmgr");
    }

    // Capture the git SHA at build time so /health can confirm the
    // running process matches what the deploy timer last saw. Falls
    // back to "unknown" outside a git checkout (e.g. shallow CI
    // clones, source tarballs) so the build never fails on this.
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=VORTEX_BENCH_BUILD_SHA={sha}");

    // HEAD covers the common deploy.sh path
    // (`git checkout --force --detach <sha>`); refs/heads/* covers
    // local branches if anyone runs the binary from a checked-out
    // branch. Both are no-ops if the file doesn't exist.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
}
