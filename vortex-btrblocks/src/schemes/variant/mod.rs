// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variant compression schemes.

mod json_to_variant;

pub use json_to_variant::JsonToVariantScheme;

#[cfg(test)]
mod tests;
