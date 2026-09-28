// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

fn main() {
    vortex_build::proto().compile(&["dtype.proto", "scalar.proto", "expr.proto"]);
}
