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
//!
//! The MSP is a signed integer of at most 64 bits, and it may use a narrower dtype when its values
//! fit. Every lower part has the dtype `u64`. An array has at most as many lower parts as the
//! canonical width of its decimal dtype holds, so the assembled values are never wider than that
//! width.
//!
//! Decoding returns the canonical decimal array, or a decimal [`Widen`] over a child at a smaller
//! precision with a narrower canonical width. The `Widen` lets a narrow MSP be read without a
//! widening copy, and only an export to the canonical width pays for that copy. See
//! [`assemble_decimal`] for the rules that pick the result.
//!
//! [`assemble_decimal`]: assemble::assemble_decimal
//! [`Widen`]: vortex_array::arrays::Widen

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;

mod array;
pub use array::*;
mod assemble;
pub(crate) mod compute;
mod plugin;
pub use plugin::DecimalBytePartsPlugin;
pub use plugin::DecimalBytePartsV2Metadata;
pub use plugin::decimal_byte_parts_v1_id;
pub use plugin::decimal_byte_parts_v2_id;
#[cfg(test)]
mod prop_tests;
mod rules;
mod split;
pub use split::DecimalParts;
pub use split::lower_part_count;
pub use split::split_decimal;
#[cfg(test)]
mod testing;

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
const MAX_LOWER_PARTS: usize = MAX_I256_LOWER_PARTS;

/// Number of bits stored in each lower part.
const LOWER_PART_BITS: usize = 64;

/// The dtype of every lower part.
const LOWER_PART_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);
