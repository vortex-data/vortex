// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Serialized metadata for TurboQuant encoding: a single byte holding the `bit_width` (0-8).
///
/// All other fields (dimension, element type) are derived from the dtype and children.
/// A `bit_width` of 0 indicates a degenerate empty array.
#[derive(Clone, Debug)]
pub struct TurboQuantMetadata {
    /// MSE bits per coordinate (0 for degenerate empty arrays, 1-8 otherwise).
    pub bit_width: u8,
}
