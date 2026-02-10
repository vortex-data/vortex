// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;
use vortex_array::builders::builder_with_capacity;
use vortex_array::expr::stats::Stat;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;
use vortex_scalar::StringLike;

pub const MAX_IS_TRUNCATED: &str = "max_is_truncated";
pub const MIN_IS_TRUNCATED: &str = "min_is_truncated";

pub fn stats_builder_with_capacity(
    stat: Stat,
    dtype: &DType,
    capacity: usize,
    max_length: usize,
) -> Box<dyn StatsArrayBuilder> {
    let values_builder = builder_with_capacity(dtype, capacity);
    match stat {
        Stat::Max => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<BufferString>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<ByteBuffer>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
        },
        Stat::Min => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMinBinaryStatsBuilder::<BufferString>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMinBinaryStatsBuilder::<ByteBuffer>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
        },
        _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
    }
}

/// Arrays with their associated names, reduced version of a StructArray
pub struct NamedArrays {
    pub names: Vec<FieldName>,
    pub arrays: Vec<ArrayRef>,
}

impl NamedArrays {
    pub fn all_invalid(&self) -> VortexResult<bool> {
        // by convention we assume that the first array is the one we care about for logical validity
        self.arrays[0].all_invalid()
    }
}

/// Minimal array builder interface for use by StatsTable for building stats arrays
pub trait StatsArrayBuilder: Send {
    fn stat(&self) -> Stat;

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()>;

    fn append_null(&mut self);

    fn finish(&mut self) -> NamedArrays;
}

pub struct StatNameArrayBuilder {
    stat: Stat,
    builder: Box<dyn ArrayBuilder>,
}

impl StatNameArrayBuilder {
    pub fn new(stat: Stat, builder: Box<dyn ArrayBuilder>) -> Self {
        Self { stat, builder }
    }
}

impl StatsArrayBuilder for StatNameArrayBuilder {
    fn stat(&self) -> Stat {
        self.stat
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        self.builder.append_scalar(&value)
    }

    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(&mut self) -> NamedArrays {
        let array = self.builder.finish();
        let len = array.len();
        match self.stat {
            Stat::Max => NamedArrays {
                names: vec![self.stat.name().into(), MAX_IS_TRUNCATED.into()],
                arrays: vec![array, ConstantArray::new(false, len).into_array()],
            },
            Stat::Min => NamedArrays {
                names: vec![self.stat.name().into(), MIN_IS_TRUNCATED.into()],
                arrays: vec![array, ConstantArray::new(false, len).into_array()],
            },
            _ => NamedArrays {
                names: vec![self.stat.name().into()],
                arrays: vec![array],
            },
        }
    }
}

struct TruncatedMaxBinaryStatsBuilder<T: ScalarTruncation> {
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    max_value_length: usize,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedMaxBinaryStatsBuilder<T> {
    pub fn new(
        values: Box<dyn ArrayBuilder>,
        is_truncated: BoolBuilder,
        max_value_length: usize,
    ) -> Self {
        Self {
            values,
            is_truncated,
            max_value_length,
            _marker: PhantomData,
        }
    }
}

struct TruncatedMinBinaryStatsBuilder<T: ScalarTruncation> {
    values: Box<dyn ArrayBuilder>,
    is_truncated: BoolBuilder,
    max_value_length: usize,
    _marker: PhantomData<T>,
}

impl<T: ScalarTruncation> TruncatedMinBinaryStatsBuilder<T> {
    pub fn new(
        values: Box<dyn ArrayBuilder>,
        is_truncated: BoolBuilder,
        max_value_length: usize,
    ) -> Self {
        Self {
            values,
            is_truncated,
            max_value_length,
            _marker: PhantomData,
        }
    }
}

pub trait ScalarTruncation: Send + Sized {
    fn from_scalar(value: Scalar) -> VortexResult<Self>;

    fn len(&self) -> usize;

    fn into_scalar(self, nullability: Nullability) -> Scalar;

    fn upper_bound(self, max_length: usize) -> Option<Self>;

    fn lower_bound(self, max_length: usize) -> Self;
}

impl ScalarTruncation for ByteBuffer {
    fn from_scalar(value: Scalar) -> VortexResult<Self> {
        value
            .into_value()
            .map(|b| b.into_binary())
            .ok_or_else(|| vortex_err!("Expected binary scalar"))
    }

    fn len(&self) -> usize {
        ByteBuffer::len(self)
    }

    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::binary(self, nullability)
    }

    /// Constructs the next [`Scalar`] at most `max_length` bytes that's lexicographically greater
    /// than this.
    ///
    /// Returns `None` if the value is null or if constructing a greater value would overflow.
    fn upper_bound(self, max_length: usize) -> Option<Self> {
        let sliced = self.slice(0..max_length);
        let mut sliced_mut = sliced.into_mut();
        for b in sliced_mut.iter_mut().rev() {
            let (incr, overflow) = b.overflowing_add(1);
            *b = incr;
            if !overflow {
                return Some(sliced_mut.freeze());
            }
        }
        None
    }

    /// Construct a [`ByteBuffer`] at most `max_length` in size that's less than or equal to
    /// ourselves.
    fn lower_bound(self, max_length: usize) -> Self {
        self.slice(0..max_length)
    }
}

impl ScalarTruncation for BufferString {
    fn from_scalar(value: Scalar) -> VortexResult<Self> {
        value
            .into_value()
            .map(|b| b.into_utf8())
            .ok_or_else(|| vortex_err!("Expected binary scalar"))
    }

    fn len(&self) -> usize {
        self.inner().len()
    }

    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::utf8(self, nullability)
    }

    /// Constructs the next [`BufferString`] at most `max_length` bytes that's lexicographically greater
    /// than this.
    ///
    /// Returns `None` if the value is null or if constructing a greater value would overflow.
    fn upper_bound(self, max_length: usize) -> Option<Self> {
        let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
            .rfind(|p| self.is_char_boundary(*p))
            .vortex_expect("Failed to find utf8 character boundary");

        // SAFETY: we slice to a char boundary so the sliced range contains valid UTF-8.
        let sliced =
            unsafe { BufferString::new_unchecked(self.into_inner().slice(..utf8_split_pos)) };
        sliced.increment().ok()
    }

    /// Construct a [`BufferString`] at most `max_length` in size that's less than or equal to
    /// ourselves.
    ///
    fn lower_bound(self, max_length: usize) -> Self {
        // UTF-8 characters are at most 4 bytes. Since we know that `BufferString` is
        // valid UTF-8, we must have a valid character boundary.
        let utf8_split_pos = (max_length.saturating_sub(3)..=max_length)
            .rfind(|p| self.is_char_boundary(*p))
            .vortex_expect("Failed to find utf8 character boundary");

        unsafe { BufferString::new_unchecked(self.into_inner().slice(..utf8_split_pos)) }
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMaxBinaryStatsBuilder<T> {
    fn stat(&self) -> Stat {
        Stat::Max
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        let (value, truncated) = upper_bound(
            if value.is_null() {
                None
            } else {
                Some(T::from_scalar(value)?)
            },
            self.max_value_length,
            nullability,
        );

        if let Some(upper_bound) = value {
            self.values.append_scalar(&upper_bound)?;
            self.is_truncated.append_value(truncated);
        } else {
            self.append_null()
        }
        Ok(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self.values.as_mut());
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![Stat::Max.name().into(), MAX_IS_TRUNCATED.into()],
            arrays: vec![
                ArrayBuilder::finish(self.values.as_mut()),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMinBinaryStatsBuilder<T> {
    fn stat(&self) -> Stat {
        Stat::Min
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let nullability = value.dtype().nullability();
        let (value, truncated) = lower_bound(
            if value.is_null() {
                None
            } else {
                Some(T::from_scalar(value)?)
            },
            self.max_value_length,
            nullability,
        );

        if let Some(lower_bound) = value {
            self.values.append_scalar(&lower_bound)?;
            self.is_truncated.append_value(truncated);
        } else {
            self.append_null()
        }
        Ok(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self.values.as_mut());
        self.is_truncated.append_value(false);
    }

    fn finish(&mut self) -> NamedArrays {
        NamedArrays {
            names: vec![Stat::Min.name().into(), MIN_IS_TRUNCATED.into()],
            arrays: vec![
                ArrayBuilder::finish(self.values.as_mut()),
                ArrayBuilder::finish(&mut self.is_truncated),
            ],
        }
    }
}

pub fn lower_bound(
    value: Option<impl ScalarTruncation>,
    max_length: usize,
    nullability: Nullability,
) -> (Option<Scalar>, bool) {
    match value {
        None => (None, false),
        Some(v) => {
            if v.len() > max_length {
                (
                    Some(v.lower_bound(max_length).into_scalar(nullability)),
                    true,
                )
            } else {
                (Some(v.into_scalar(nullability)), false)
            }
        }
    }
}

pub fn upper_bound(
    value: Option<impl ScalarTruncation>,
    max_length: usize,
    nullability: Nullability,
) -> (Option<Scalar>, bool) {
    match value {
        None => (None, false),
        Some(v) => {
            if v.len() > max_length {
                (
                    v.upper_bound(max_length)
                        .map(|v| v.into_scalar(nullability)),
                    true,
                )
            } else {
                (Some(v.into_scalar(nullability)), false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferString;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;

    use crate::layouts::zoned::builder::ScalarTruncation;
    use crate::layouts::zoned::lower_bound;
    use crate::layouts::zoned::upper_bound;

    #[test]
    fn binary_lower_bound() {
        let binary = buffer![0u8, 5, 47, 33, 129];
        let expected = buffer![0u8, 5];
        assert_eq!(binary.lower_bound(2), expected,);
    }

    #[test]
    fn binary_upper_bound() {
        let binary = buffer![0u8, 5, 255, 234, 23];
        let expected = buffer![0u8, 6, 0];
        assert_eq!(binary.upper_bound(3).unwrap(), expected,);
    }

    #[test]
    fn binary_upper_bound_overflow() {
        let binary = buffer![255u8, 255, 255];
        assert!(binary.upper_bound(2).is_none());
    }

    #[test]
    fn binary_upper_bound_null() {
        assert!(
            upper_bound(Option::<ByteBuffer>::None, 10, Nullability::Nullable)
                .0
                .is_none()
        );
    }

    #[test]
    fn binary_lower_bound_null() {
        assert!(
            lower_bound(Option::<ByteBuffer>::None, 10, Nullability::Nullable)
                .0
                .is_none()
        );
    }

    #[test]
    fn utf8_lower_bound() {
        let utf8 = BufferString::from("snowman⛄️snowman");
        let expected = BufferString::from("snowman");
        assert_eq!(utf8.lower_bound(9), expected);
    }

    #[test]
    fn utf8_upper_bound() {
        let utf8 = BufferString::from("char🪩");
        let expected = BufferString::from("chas");
        assert_eq!(utf8.upper_bound(5).unwrap(), expected);
    }

    #[test]
    fn utf8_upper_bound_overflow() {
        let utf8 = BufferString::from("🂑🂒🂓");
        assert!(utf8.upper_bound(2).is_none());
    }

    #[test]
    fn utf8_upper_bound_null() {
        assert!(
            upper_bound(Option::<BufferString>::None, 10, Nullability::Nullable)
                .0
                .is_none()
        );
    }

    #[test]
    fn utf8_lower_bound_null() {
        assert!(
            lower_bound(Option::<BufferString>::None, 10, Nullability::Nullable)
                .0
                .is_none()
        );
    }
}
