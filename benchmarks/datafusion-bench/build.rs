// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    // The linker flag is for the target, so check the target OS and not the host OS.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        custom_labels::build::emit_build_instructions();
    }
}
