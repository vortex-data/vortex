// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    // Rerun if git HEAD changes
    println!("cargo::rerun-if-changed=../.git/HEAD");
    println!("cargo::rerun-if-changed=../.git/refs/tags");

    // Try to get version from git describe
    if let Ok(output) = std::process::Command::new("git")
        .args(["describe", "--tags", "--always"])
        .output()
        && output.status.success()
    {
        let version = String::from_utf8_lossy(&output.stdout);
        println!("cargo::rustc-env=VX_VERSION={}", version.trim());
    }
}
