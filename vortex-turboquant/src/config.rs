// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Minimum vector dimension for TurboQuant encoding.
///
/// Not a theoretical minimum, just a practical floor to limit total distortion. The minimum
/// per-block width [`MIN_BLOCK_SIZE`] is defined to equal this, so the smallest valid input is a
/// single minimum-width block; the two floors are intentionally tied to the same value.
pub(crate) const MIN_DIMENSION: u32 = 64;

/// Minimum power-of-two block size.
pub(crate) const MIN_BLOCK_SIZE: u32 = MIN_DIMENSION;

/// Maximum supported number of bits per quantized coordinate.
pub(crate) const MAX_BIT_WIDTH: u8 = 8;

/// Configuration for lossy TurboQuant encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TurboQuantConfig {
    bit_width: u8,
    seed: u64,
    num_rounds: u8,
    block_sizes: Option<Vec<u32>>,
}

impl Default for TurboQuantConfig {
    /// Defaults to 8 bits per coordinate, seed 42, 3 SORF rounds, and the encode-time default
    /// block decomposition.
    fn default() -> Self {
        Self {
            bit_width: MAX_BIT_WIDTH,
            seed: 42,
            num_rounds: 3,
            block_sizes: None,
        }
    }
}

impl TurboQuantConfig {
    /// Build a TurboQuant configuration.
    ///
    /// When `block_sizes` is `None`, the encoder defaults to a single power-of-two block covering
    /// the full dimension. When `Some`, the blocks are validated (non-empty, power-of-two, greater
    /// than `MIN_BLOCK_SIZE`, and sum covers all dimensions).
    ///
    /// # Errors
    ///
    /// Returns an error if `bit_width` is outside `1..=8`, `num_rounds` is zero, or the supplied
    /// `block_sizes` violate any of the dimension-independent rules.
    pub fn try_new(
        bit_width: u8,
        seed: u64,
        num_rounds: u8,
        block_sizes: Option<Vec<u32>>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (1..=MAX_BIT_WIDTH).contains(&bit_width),
            "TurboQuant bit_width must be 1-{MAX_BIT_WIDTH}, got {bit_width}",
        );
        vortex_ensure!(
            num_rounds > 0,
            "TurboQuant num_rounds must be > 0, got {num_rounds}"
        );

        if let Some(block_sizes) = block_sizes.as_deref() {
            validate_block_shape(block_sizes)?;
        }

        Ok(Self {
            bit_width,
            seed,
            num_rounds,
            block_sizes,
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

    /// User-supplied power-of-two block decomposition, if any. `None` defers block resolution to
    /// the encoder, which then picks a single block of the dimension rounded up to a power of two.
    pub fn block_sizes(&self) -> Option<&[u32]> {
        self.block_sizes.as_deref()
    }
}

impl fmt::Display for TurboQuantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bit_width: {}, seed: {}, num_rounds: {}, block_sizes: ",
            self.bit_width, self.seed, self.num_rounds
        )?;

        match self.block_sizes.as_deref() {
            None => write!(f, "None"),
            Some(block_sizes) => {
                write!(f, "Some([")?;
                for (index, block) in block_sizes.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{block}")?;
                }
                write!(f, "])")
            }
        }
    }
}

/// Validate the dimension-independent block-shape rules: non-empty, power-of-two, each block at
/// least `MIN_BLOCK_SIZE`.
pub(crate) fn validate_block_shape(block_sizes: &[u32]) -> VortexResult<()> {
    vortex_ensure!(
        !block_sizes.is_empty(),
        "TurboQuant block_sizes must be non-empty"
    );

    for (index, &block) in block_sizes.iter().enumerate() {
        vortex_ensure!(
            block >= MIN_BLOCK_SIZE,
            "TurboQuant block {index} must be >= {MIN_BLOCK_SIZE}, got {block}"
        );
        vortex_ensure!(
            block.is_power_of_two(),
            "TurboQuant block {index} must be a power of two, got {block}"
        );
    }
    Ok(())
}

/// Validate the dimension-dependent rule that the resolved blocks cover every dimension. The
/// encoder (`resolve_block_sizes`) and metadata validation (`validate_tq_metadata`) both call this.
pub(crate) fn validate_block_sum(block_sizes: &[u32], dimensions: u32) -> VortexResult<()> {
    let sum: u64 = block_sizes.iter().map(|&block| block as u64).sum();
    vortex_ensure!(
        sum >= dimensions as u64,
        "TurboQuant block_sizes sum {sum} must be >= dimensions {dimensions}"
    );
    Ok(())
}
