// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal byte-parts encoding.
//!
//! A `DecimalByteParts` array stores each value as a signed most significant part (MSP)
//! followed by `k` unsigned 64-bit lower parts ordered most significant first. The encoded
//! value is
//!
//! ```text
//! msp * 2^(64k) + Σ_{i<k} lower[i] * 2^(64 * (k - 1 - i))
//! ```
//!
//! This is exactly the two's complement bit pattern of the decimal value cut on 64-bit
//! boundaries.
//! Each part may use a narrower integer dtype when its values fit; the word boundaries stay fixed.

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;

mod array;
mod assemble;
pub(crate) mod compute;
mod plugin;
#[cfg(test)]
mod prop_tests;
mod rules;
mod split;
#[cfg(test)]
mod testing;

pub use array::*;
pub use plugin::DecimalBytePartsPlugin;
pub use plugin::DecimalBytePartsV2Metadata;
pub use plugin::decimal_byte_parts_v1_id;
pub use plugin::decimal_byte_parts_v2_id;
pub use split::DecimalParts;
pub use split::split_decimal;

#[doc(hidden)]
pub mod _benchmarking {
    pub use super::assemble::assemble_decimal;
    pub use super::assemble::assemble_wide_decimal;
    pub use super::split::i128_to_parts;
    pub use super::split::i256_to_parts;
    pub use super::split::split_wide;
}

/// The maximum number of 64-bit lower parts an encoded `i128` decimal can carry.
const MAX_I128_LOWER_PARTS: usize = 1;

/// The maximum number of 64-bit lower parts an encoded `i256` decimal can carry.
const MAX_I256_LOWER_PARTS: usize = 3;

/// The maximum number of 64-bit lower parts an encoded decimal can carry.
pub const MAX_LOWER_PARTS: usize = MAX_I256_LOWER_PARTS;

/// Number of bits stored in each lower part.
const LOWER_PART_BITS: usize = 64;

/// Dtype of the lower parts produced by splitting, before any narrowing.
const LOWER_PART_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);
