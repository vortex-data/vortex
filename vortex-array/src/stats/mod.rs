//! Traits and utilities to compute and access array statistics.

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use arrow_buffer::bit_iterator::BitIterator;
use arrow_buffer::{BooleanBufferBuilder, MutableBuffer};
use enum_iterator::{Sequence, last};
use log::debug;
use num_enum::{IntoPrimitive, TryFromPrimitive};
pub use stats_set::*;
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, PType};

mod array;
mod bound;
pub mod flatbuffers;
mod precision;
mod stat_bound;
mod stats_set;
mod traits;

pub use array::*;
pub use bound::{LowerBound, UpperBound};
pub use precision::Precision;
pub use stat_bound::*;
pub use traits::*;
use vortex_error::VortexExpect;

/// Statistics that are used for pruning files (i.e., we want to ensure they are computed when compressing/writing).
/// Sum is included for boolean arrays.
pub const PRUNING_STATS: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::NullCount,
    Stat::NaNCount,
];

/// Stats to keep when serializing arrays to layouts
pub const STATS_TO_WRITE: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::NullCount,
    Stat::NaNCount,
    Stat::Sum,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
    Stat::UncompressedSizeInBytes,
];

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Sequence,
    IntoPrimitive,
    TryFromPrimitive,
)]
#[repr(u8)]
pub enum Stat {
    /// Whether all values are the same (nulls are not equal to other non-null values,
    /// so this is true iff all values are null or all values are the same non-null value)
    IsConstant = 0,
    /// Whether the non-null values in the array are sorted (i.e., we skip nulls)
    IsSorted = 1,
    /// Whether the non-null values in the array are strictly sorted (i.e., sorted with no duplicates)
    IsStrictSorted = 2,
    /// The maximum value in the array (ignoring nulls, unless all values are null)
    Max = 3,
    /// The minimum value in the array (ignoring nulls, unless all values are null)
    Min = 4,
    /// The sum of the non-null values of the array.
    Sum = 5,
    /// The number of null values in the array
    NullCount = 6,
    /// The uncompressed size of the array in bytes
    UncompressedSizeInBytes = 7,
    /// The number of NaN values in the array
    NaNCount = 8,
}

/// These structs allow the extraction of the bound from the `Precision` value.
/// They tie together the Stat and the StatBound, which allows the bound to be extracted.
pub struct Max;
pub struct Min;
pub struct Sum;
pub struct IsConstant;
pub struct IsSorted;
pub struct IsStrictSorted;
pub struct NullCount;
pub struct UncompressedSizeInBytes;
pub struct NaNCount;

impl StatType<bool> for IsConstant {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsConstant;
}

impl StatType<bool> for IsSorted {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsSorted;
}

impl StatType<bool> for IsStrictSorted {
    type Bound = Precision<bool>;

    const STAT: Stat = Stat::IsStrictSorted;
}

impl<T: PartialOrd + Clone> StatType<T> for NullCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::NullCount;
}

impl<T: PartialOrd + Clone> StatType<T> for UncompressedSizeInBytes {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::UncompressedSizeInBytes;
}

impl<T: PartialOrd + Clone + Debug> StatType<T> for Max {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::Max;
}

impl<T: PartialOrd + Clone + Debug> StatType<T> for Min {
    type Bound = LowerBound<T>;

    const STAT: Stat = Stat::Min;
}

impl<T: PartialOrd + Clone + Debug> StatType<T> for Sum {
    type Bound = Precision<T>;

    const STAT: Stat = Stat::Sum;
}

impl<T: PartialOrd + Clone> StatType<T> for NaNCount {
    type Bound = UpperBound<T>;

    const STAT: Stat = Stat::NaNCount;
}

impl Stat {
    /// Whether the statistic is commutative (i.e., whether merging can be done independently of ordering)
    /// e.g., min/max are commutative, but is_sorted is not
    pub fn is_commutative(&self) -> bool {
        // NOTE: we prefer this syntax to force a compile error if we add a new stat
        match self {
            Self::IsConstant
            | Self::Max
            | Self::Min
            | Self::NullCount
            | Self::Sum
            | Self::NaNCount
            | Self::UncompressedSizeInBytes => true,
            Self::IsSorted | Self::IsStrictSorted => false,
        }
    }

    /// Whether the statistic has the same dtype as the array it's computed on
    pub fn has_same_dtype_as_array(&self) -> bool {
        matches!(self, Stat::Min | Stat::Max)
    }

    /// Return the [`DType`] of the statistic scalar assuming the array is of the given [`DType`].
    pub fn dtype(&self, data_type: &DType) -> Option<DType> {
        Some(match self {
            Self::IsConstant => DType::Bool(NonNullable),
            Self::IsSorted => DType::Bool(NonNullable),
            Self::IsStrictSorted => DType::Bool(NonNullable),
            Self::Max => data_type.clone(),
            Self::Min => data_type.clone(),
            Self::NullCount => DType::Primitive(PType::U64, NonNullable),
            Self::UncompressedSizeInBytes => DType::Primitive(PType::U64, NonNullable),
            Self::NaNCount => match data_type {
                DType::Primitive(ptype, ..) if ptype.is_float() => {
                    DType::Primitive(PType::U64, NonNullable)
                }
                // Any other type does not support NaN count
                _ => return None,
            },
            Self::Sum => {
                // Any array that cannot be summed has a sum DType of null.
                // Any array that can be summed, but overflows, has a sum _value_ of null.
                // Therefore, we make integer sum stats nullable.
                match data_type {
                    DType::Bool(_) => DType::Primitive(PType::U64, Nullable),
                    DType::Primitive(ptype, _) => match ptype {
                        PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                            DType::Primitive(PType::U64, Nullable)
                        }
                        PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                            DType::Primitive(PType::I64, Nullable)
                        }
                        PType::F16 | PType::F32 | PType::F64 => {
                            // Float sums cannot overflow, so it's non-nullable
                            DType::Primitive(PType::F64, NonNullable)
                        }
                    },
                    DType::Extension(ext_dtype) => self.dtype(ext_dtype.storage_dtype())?,
                    // Unsupported types
                    DType::Null
                    // TODO(aduffy): implement more stats for Decimal
                    | DType::Decimal(..)
                    | DType::Utf8(_)
                    | DType::Binary(_)
                    | DType::Struct(..)
                    | DType::List(..) => return None,
                }
            }
        })
    }

    pub fn name(&self) -> &str {
        match self {
            Self::IsConstant => "is_constant",
            Self::IsSorted => "is_sorted",
            Self::IsStrictSorted => "is_strict_sorted",
            Self::Max => "max",
            Self::Min => "min",
            Self::NullCount => "null_count",
            Self::UncompressedSizeInBytes => "uncompressed_size_in_bytes",
            Self::Sum => "sum",
            Self::NaNCount => "nan_count",
        }
    }
}

pub fn as_stat_bitset_bytes(stats: &[Stat]) -> Vec<u8> {
    let max_stat = u8::from(last::<Stat>().vortex_expect("last stat")) as usize + 1;
    // TODO(ngates): use vortex-buffer::BitBuffer
    let mut stat_bitset = BooleanBufferBuilder::new_from_buffer(
        MutableBuffer::from_len_zeroed(max_stat.div_ceil(8)),
        max_stat,
    );
    for stat in stats {
        stat_bitset.set_bit(u8::from(*stat) as usize, true);
    }

    stat_bitset
        .finish()
        .into_inner()
        .into_vec()
        .unwrap_or_else(|b| b.to_vec())
}

pub fn stats_from_bitset_bytes(bytes: &[u8]) -> Vec<Stat> {
    BitIterator::new(bytes, 0, bytes.len() * 8)
        .enumerate()
        .filter_map(|(i, b)| b.then_some(i))
        // Filter out indices failing conversion, these are stats written by newer version of library
        .filter_map(|i| {
            let Ok(stat) = u8::try_from(i) else {
                debug!("invalid stat encountered: {i}");
                return None;
            };
            Stat::try_from(stat).ok()
        })
        .collect::<Vec<_>>()
}

impl Display for Stat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod test {
    use enum_iterator::all;

    use crate::arrays::PrimitiveArray;
    use crate::stats::Stat;

    #[test]
    fn min_of_nulls_is_not_panic() {
        let min = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None])
            .statistics()
            .compute_as::<i64>(Stat::Min);

        assert_eq!(min, None);
    }

    #[test]
    fn has_same_dtype_as_array() {
        assert!(Stat::Min.has_same_dtype_as_array());
        assert!(Stat::Max.has_same_dtype_as_array());
        for stat in all::<Stat>().filter(|s| !matches!(s, Stat::Min | Stat::Max)) {
            assert!(!stat.has_same_dtype_as_array());
        }
    }
}
