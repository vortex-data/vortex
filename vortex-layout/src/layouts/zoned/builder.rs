// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_array::arrays::ConstantArray;
use vortex_array::builders::{ArrayBuilder, BoolBuilder, builder_with_capacity};
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldName, Nullability};
use vortex_error::VortexResult;
use vortex_scalar::{BinaryScalar, Scalar, Utf8Scalar};

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
            DType::Utf8(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<Utf8Scalar>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMaxBinaryStatsBuilder::<BinaryScalar>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            _ => Box::new(StatNameArrayBuilder::new(stat, values_builder)),
        },
        Stat::Min => match dtype {
            DType::Utf8(_) => Box::new(TruncatedMinBinaryStatsBuilder::<Utf8Scalar>::new(
                values_builder,
                BoolBuilder::with_capacity(Nullability::NonNullable, capacity),
                max_length,
            )),
            DType::Binary(_) => Box::new(TruncatedMinBinaryStatsBuilder::<BinaryScalar>::new(
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
    pub fn all_invalid(&self) -> bool {
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
    fn from_scalar(value: &Scalar) -> VortexResult<impl ScalarTruncation>;

    fn len(&self) -> Option<usize>;

    fn into_scalar(self) -> Scalar;

    fn upper_bound(self, max_length: usize) -> Option<Self>;

    fn lower_bound(self, max_length: usize) -> Self;
}

impl ScalarTruncation for BinaryScalar<'_> {
    fn from_scalar(value: &Scalar) -> VortexResult<impl ScalarTruncation> {
        BinaryScalar::try_from(value)
    }

    fn len(&self) -> Option<usize> {
        self.len()
    }

    fn into_scalar(self) -> Scalar {
        self.value()
            .map(|b| Scalar::binary(b, self.dtype().nullability()))
            .unwrap_or_else(|| Scalar::null(self.dtype().clone()))
    }

    fn upper_bound(self, max_length: usize) -> Option<Self> {
        self.upper_bound(max_length)
    }

    fn lower_bound(self, max_length: usize) -> Self {
        self.lower_bound(max_length)
    }
}

impl ScalarTruncation for Utf8Scalar<'_> {
    fn from_scalar(value: &Scalar) -> VortexResult<impl ScalarTruncation> {
        Utf8Scalar::try_from(value)
    }

    fn len(&self) -> Option<usize> {
        self.len()
    }

    fn into_scalar(self) -> Scalar {
        self.value()
            .map(|b| Scalar::utf8(b, self.dtype().nullability()))
            .unwrap_or_else(|| Scalar::null(self.dtype().clone()))
    }

    fn upper_bound(self, max_length: usize) -> Option<Self> {
        self.upper_bound(max_length)
    }

    fn lower_bound(self, max_length: usize) -> Self {
        self.lower_bound(max_length)
    }
}

impl<T: ScalarTruncation> StatsArrayBuilder for TruncatedMaxBinaryStatsBuilder<T> {
    fn stat(&self) -> Stat {
        Stat::Max
    }

    fn append_scalar(&mut self, value: Scalar) -> VortexResult<()> {
        let (value, truncated) = upper_bound(T::from_scalar(&value)?, self.max_value_length);

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
        let (value, truncated) = lower_bound(T::from_scalar(&value)?, self.max_value_length);
        self.values.append_scalar(&value)?;
        self.is_truncated.append_value(truncated);
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

pub fn lower_bound(value: impl ScalarTruncation, max_length: usize) -> (Scalar, bool) {
    if value.len().unwrap_or(0) > max_length {
        (value.lower_bound(max_length).into_scalar(), true)
    } else {
        (value.into_scalar(), false)
    }
}

pub fn upper_bound(value: impl ScalarTruncation, max_length: usize) -> (Option<Scalar>, bool) {
    if value.len().unwrap_or(0) > max_length {
        (value.upper_bound(max_length).map(|v| v.into_scalar()), true)
    } else {
        (Some(value.into_scalar()), false)
    }
}
