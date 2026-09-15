// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Force both native objects into the host link, which has no instrumentation runtimes.
    assert!(native_helper::value() > 10);
    let version = fs::read_to_string("header-version")?;
    fs::create_dir_all("cinclude")?;
    fs::write(
        "cinclude/vortex.h",
        format!("#define VORTEX_HEADER_VERSION {}\n", version.trim()),
    )?;
    println!("cargo:rerun-if-changed=header-version");
    Ok(())
}
