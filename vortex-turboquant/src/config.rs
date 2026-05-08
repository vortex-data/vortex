// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Minimum vector dimension for TurboQuant encoding.
///
/// Note that this is not a theoretical minimum, it is mostly a practical one to limit the total
/// amount of distortion.
pub(crate) const MIN_DIMENSION: u32 = 128;

/// Maximum supported number of bits per quantized coordinate.
pub(crate) const MAX_BIT_WIDTH: u8 = 8;

/// Configuration for lossy TurboQuant encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TurboQuantConfig {
    bit_width: u8,
    seed: u64,
    num_rounds: u8,
}

impl TurboQuantConfig {
    /// Build a TurboQuant configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `bit_width` is outside `1..=8` or `num_rounds` is zero.
    pub fn try_new(bit_width: u8, seed: u64, num_rounds: u8) -> VortexResult<Self> {
        vortex_ensure!(
            (1..=MAX_BIT_WIDTH).contains(&bit_width),
            "TurboQuant bit_width must be 1-{MAX_BIT_WIDTH}, got {bit_width}",
        );
        vortex_ensure!(
            num_rounds > 0,
            "TurboQuant num_rounds must be > 0, got {num_rounds}"
        );

        Ok(Self {
            bit_width,
            seed,
            num_rounds,
        })
    }

    /// Bits per coordinate in the scalar quantizer codebook.
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Seed used to derive the deterministic SORF transform.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of sign-diagonal plus Walsh-Hadamard rounds in the SORF transform.
    pub fn num_rounds(&self) -> u8 {
        self.num_rounds
    }
}

impl Default for TurboQuantConfig {
    /// Defaults to 8 bits per coordinate, seed 42, and 3 SORF rounds.
    fn default() -> Self {
        Self {
            bit_width: MAX_BIT_WIDTH,
            seed: 42,
            num_rounds: 3,
        }
    }
}

impl fmt::Display for TurboQuantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bit_width: {}, seed: {}, num_rounds: {}",
            self.bit_width, self.seed, self.num_rounds
        )
    }
}
