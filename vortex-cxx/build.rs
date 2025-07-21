// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    let mut _builder = cxx_build::bridge("src/lib.rs");

    println!("cargo:rerun-if-changed=src/");
}
