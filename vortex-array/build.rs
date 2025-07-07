// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[rustversion::nightly]
fn main() {
    println!("cargo:rustc-cfg=nightly");
}

#[rustversion::not(nightly)]
fn main() {}
