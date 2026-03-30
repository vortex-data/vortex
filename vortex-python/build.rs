// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    #[cfg(target_os = "macos")]
    {
        // For pyo3 to successfully link on macOS.
        // See https://stackoverflow.com/a/77382609
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
