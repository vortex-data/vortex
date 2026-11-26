// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    // `custom-labels` is only a build dependency on Linux hosts, and its linker flag is for the
    // target, so both have to be Linux.
    #[cfg(target_os = "linux")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        custom_labels::build::emit_build_instructions();
    }
}
