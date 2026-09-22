// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    vortex_build::flatbuffers().compile(&["vortex-layout/layout.fbs"]);
}
