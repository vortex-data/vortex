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
//! boundaries. [`DecimalBytePartsData`] owns the rules for the part dtypes and count, and
//! [`assemble_decimal`] owns canonical decoding. The [crate documentation](crate) explains the
//! logical MSP dtype and why narrow storage needs no scan.
//!
//! [`assemble_decimal`]: assemble::assemble_decimal

use vortex_array::ArrayRef;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;

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

/// The logical signed MSP type for a decimal precision, capped at `i64`.
fn msp_ptype(decimal_dtype: DecimalDType) -> PType {
    match decimal_dtype.canonical_values_type() {
        DecimalType::I8 => PType::I8,
        DecimalType::I16 => PType::I16,
        DecimalType::I32 => PType::I32,
        DecimalType::I64 | DecimalType::I128 | DecimalType::I256 => PType::I64,
    }
}

/// Returns the narrow integer child of a [`Widen`] MSP, or the MSP itself when it is not a
/// [`Widen`].
fn stored_msp(msp: &ArrayRef) -> ArrayRef {
    match msp.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => msp.clone(),
    }
}

/// The largest number of lower parts that assemble to at most the canonical width of
/// `decimal_dtype`.
///
/// One lower part assembles to `i128`, and two or three assemble to `i256`. An MSP is at most
/// `i64`, so a canonical width of `i64` or less allows no lower parts. With this bound, decoding
/// never narrows values that it assembled from lower parts.
fn max_lower_parts(decimal_dtype: DecimalDType) -> usize {
    match decimal_dtype.canonical_values_type() {
        DecimalType::I8 | DecimalType::I16 | DecimalType::I32 | DecimalType::I64 => 0,
        DecimalType::I128 => MAX_I128_LOWER_PARTS,
        DecimalType::I256 => MAX_I256_LOWER_PARTS,
    }
}

mod array;
pub use array::*;

mod assemble;
pub(crate) mod compute;

mod plugin;
pub use plugin::DecimalBytePartsPlugin;
pub use plugin::DecimalBytePartsV2Metadata;
pub use plugin::decimal_byte_parts_v1_id;
pub use plugin::decimal_byte_parts_v2_id;

mod rules;

mod split;
pub use split::DecimalParts;
pub use split::lower_part_count;
pub use split::split_decimal;

#[doc(hidden)]
pub mod _benchmarking {
    //! Internal entry points for the assembly and split microbenchmarks.
    pub use super::assemble::assemble_decimal;
    pub use super::assemble::assemble_wide_decimal;
    pub use super::split::i128_to_parts;
    pub use super::split::i256_to_parts;
    pub use super::split::split_wide;
}

#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod testing;
