// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Arrays can provide compute implementations written in various shader languages.
///
/// These implementations operate over GPU vector inputs and return GPU vector outputs.
pub trait WGSLShader {
    fn shader_code(&self) -> String;
}
