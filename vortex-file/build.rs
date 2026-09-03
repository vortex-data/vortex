// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    vortex_build::flatbuffers()
        .depends_on("vortex-array")
        .depends_on("vortex-layout")
        .compile(&["vortex-file/footer.fbs"]);
}
