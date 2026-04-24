// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    #[cfg(not(target_os = "macos"))]
    custom_labels::build::emit_build_instructions();
}
