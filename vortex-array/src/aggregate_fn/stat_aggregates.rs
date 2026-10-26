// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Identifiers and options of the aggregate functions that back array statistics.
//!
//! [`Stat`](crate::expr::stats::Stat) names each of these aggregates by ID and options, and
//! array statistics are computed by resolving them through the session's
//! [`AggregateFnSession`](crate::aggregate_fn::session::AggregateFnSession). The
//! implementations themselves can therefore be registered from any crate.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::MAX_PRECISION;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::proto::expr as pb;

/// ID of the aggregate computing the minimum, the `Min` statistic.
pub static MIN_ID: CachedId = CachedId::new("vortex.min");
/// ID of the aggregate computing the minimum and maximum together, the `Min` and `Max` statistics
/// in one pass.
pub static MIN_MAX_ID: CachedId = CachedId::new("vortex.min_max");
/// ID of the aggregate computing the maximum, the `Max` statistic.
pub static MAX_ID: CachedId = CachedId::new("vortex.max");
/// ID of the aggregate computing the sum, the `Sum` statistic.
pub static SUM_ID: CachedId = CachedId::new("vortex.sum");
/// ID of the aggregate counting nulls, the `NullCount` statistic.
pub static NULL_COUNT_ID: CachedId = CachedId::new("vortex.null_count");
/// ID of the aggregate counting NaN values, the `NaNCount` statistic.
pub static NAN_COUNT_ID: CachedId = CachedId::new("vortex.nan_count");
/// ID of the aggregate measuring uncompressed size, the `UncompressedSizeInBytes` statistic.
pub static UNCOMPRESSED_SIZE_IN_BYTES_ID: CachedId =
    CachedId::new("vortex.uncompressed_size_in_bytes");
/// ID of the aggregate checking constancy, the `IsConstant` statistic.
pub static IS_CONSTANT_ID: CachedId = CachedId::new("vortex.is_constant");
/// ID of the aggregate checking sortedness, the `IsSorted` and `IsStrictSorted` statistics.
pub static IS_SORTED_ID: CachedId = CachedId::new("vortex.is_sorted");
/// ID of the aggregate checking that every value is null.
pub static ALL_NULL_ID: CachedId = CachedId::new("vortex.all_null");
/// ID of the aggregate checking that no value is null.
pub static ALL_NON_NULL_ID: CachedId = CachedId::new("vortex.all_non_null");
/// ID of the aggregate checking that every value is NaN.
pub static ALL_NAN_ID: CachedId = CachedId::new("vortex.all_nan");
/// ID of the aggregate checking that no value is NaN.
pub static ALL_NON_NAN_ID: CachedId = CachedId::new("vortex.all_non_nan");

/// Whether the `vortex.uncompressed_size_in_bytes` aggregate approximates the bytes an array uses or counts them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UncompressedSizePrecision {
    /// An approximation: every byte of every buffer the canonical array holds, whether or not a
    /// row uses it. Cheap, but a view array sharing or sliced from a larger buffer reports the
    /// whole buffer.
    #[default]
    Inexact,
    /// The actual number of bytes the array's rows use, which costs a pass over view and list
    /// arrays.
    Exact,
}

/// Options for the `vortex.uncompressed_size_in_bytes` aggregate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UncompressedSizeOpts {
    pub precision: UncompressedSizePrecision,
}

impl UncompressedSizeOpts {
    /// Approximate the bytes the array uses by the buffers it holds. This is the default.
    pub const fn inexact() -> Self {
        Self {
            precision: UncompressedSizePrecision::Inexact,
        }
    }

    /// Count the actual bytes the array's rows use.
    pub const fn exact() -> Self {
        Self {
            precision: UncompressedSizePrecision::Exact,
        }
    }

    pub const fn is_exact(&self) -> bool {
        matches!(self.precision, UncompressedSizePrecision::Exact)
    }

    /// Serialize these options to protobuf-encoded metadata bytes.
    pub fn serialize(&self) -> Vec<u8> {
        pb::UncompressedSizeInBytesOpts {
            exact: self.is_exact(),
        }
        .encode_to_vec()
    }

    /// Deserialize these options from protobuf-encoded metadata bytes.
    pub fn deserialize(metadata: &[u8]) -> VortexResult<Self> {
        let opts = pb::UncompressedSizeInBytesOpts::decode(metadata)?;
        Ok(if opts.exact {
            Self::exact()
        } else {
            Self::inexact()
        })
    }
}

impl Display for UncompressedSizeOpts {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Only the non-default configuration is displayed, so the default renders as
        // `vortex.uncompressed_size_in_bytes()`.
        if self.is_exact() {
            write!(f, "precision=exact")?;
        }
        Ok(())
    }
}

/// Options for the `is_sorted` aggregate function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IsSortedOptions {
    /// If true, check for strictly ascending order (no duplicates).
    pub strict: bool,
}

impl IsSortedOptions {
    /// Serialize these options to metadata bytes: one byte, `1` for strict and `0` otherwise.
    pub fn serialize(&self) -> Vec<u8> {
        vec![u8::from(self.strict)]
    }

    /// Deserialize these options from metadata bytes written by [`IsSortedOptions::serialize`].
    pub fn deserialize(metadata: &[u8]) -> VortexResult<Self> {
        match metadata {
            [0] => Ok(Self { strict: false }),
            [1] => Ok(Self { strict: true }),
            _ => vortex_bail!("invalid is_sorted options metadata {metadata:?}"),
        }
    }
}

impl Display for IsSortedOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "strict={}", self.strict)
    }
}

/// The return dtype of the `vortex.sum` aggregate over `input_dtype`, or `None` if summing that
/// dtype is unsupported.
///
/// A sum that overflows is null, so every result dtype is nullable.
pub fn sum_return_dtype(input_dtype: &DType) -> Option<DType> {
    use Nullability::Nullable;

    Some(match input_dtype {
        DType::Bool(_) => DType::Primitive(PType::U64, Nullable),
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                DType::Primitive(PType::U64, Nullable)
            }
            PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                DType::Primitive(PType::I64, Nullable)
            }
            // Float sums cannot overflow, but all-null floats still sum to null.
            PType::F16 | PType::F32 | PType::F64 => DType::Primitive(PType::F64, Nullable),
        },
        DType::Decimal(decimal_dtype, _) => {
            DType::Decimal(sum_decimal_dtype(decimal_dtype), Nullable)
        }
        _ => return None,
    })
}

// Both Spark and DataFusion use this heuristic.
// - https://github.com/apache/spark/blob/fcf636d9eb8d645c24be3db2d599aba2d7e2955a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/Sum.scala#L66
// - https://github.com/apache/datafusion/blob/4153adf2c0f6e317ef476febfdc834208bd46622/datafusion/functions-aggregate/src/sum.rs#L188
/// Returns the decimal dtype a sum over `input` widens to.
pub fn sum_decimal_dtype(input: &DecimalDType) -> DecimalDType {
    DecimalDType::new(
        u8::min(MAX_PRECISION, input.precision() + 10),
        input.scale(),
    )
}

/// Whether the uncompressed size of an array of `dtype` can be measured.
pub fn supports_uncompressed_size_in_bytes(dtype: &DType) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            supports_uncompressed_size_in_bytes(element_dtype)
        }
        DType::Map(map_dtype, _) => {
            supports_uncompressed_size_in_bytes(&map_dtype.key_dtype())
                && supports_uncompressed_size_in_bytes(&map_dtype.value_dtype())
        }
        DType::Struct(fields, _) => fields
            .fields()
            .all(|field| supports_uncompressed_size_in_bytes(&field)),
        DType::Union(variants, _) => variants
            .variants()
            .all(|variant| supports_uncompressed_size_in_bytes(&variant)),
        DType::Variant(_) => false,
        DType::Extension(ext_dtype) => {
            supports_uncompressed_size_in_bytes(ext_dtype.storage_dtype())
        }
    }
}
