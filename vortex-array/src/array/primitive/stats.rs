use core::marker::PhantomData;
use std::cmp::Ordering;
use std::mem::size_of;

use arrow_buffer::buffer::BooleanBuffer;
use itertools::{Itertools as _, MinMaxResult};
use num_traits::PrimInt;
use vortex_dtype::half::f16;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::aliases::hash_map::HashMap;
use crate::array::primitive::PrimitiveArray;
use crate::stats::{ArrayStatisticsCompute, Stat, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayDType, IntoArrayVariant};

trait PStatsType: NativePType + Into<Scalar> + BitWidth {}

impl<T: NativePType + Into<Scalar> + BitWidth> PStatsType for T {}

impl ArrayStatisticsCompute for PrimitiveArray {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        match_each_native_ptype!(self.ptype(), |$P| {
            match self.logical_validity() {
                LogicalValidity::AllValid(_) => self.maybe_null_slice::<$P>().compute_statistics(stat),
                LogicalValidity::AllInvalid(v) => Ok(StatsSet::nulls(v, self.dtype())),
                LogicalValidity::Array(a) => NullableValues(
                    self.maybe_null_slice::<$P>(),
                    &a.clone().into_bool()?.boolean_buffer(),
                )
                .compute_statistics(stat),
            }
        })
    }
}

impl<T: PStatsType> ArrayStatisticsCompute for &[T] {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        if self.is_empty() {
            return Ok(StatsSet::new());
        }

        if matches!(stat, Stat::Min | Stat::Max) {
            // this `compare` function provides a total ordering (even for NaN values)
            match self.iter().minmax_by(|a, b| a.compare(**b)) {
                MinMaxResult::NoElements => return Ok(StatsSet::new()),
                MinMaxResult::OneElement(x) => {
                    let scalar: Scalar = (*x).into();
                    return Ok(StatsSet::from(HashMap::from([
                        (Stat::Min, scalar.clone()),
                        (Stat::Max, scalar),
                        (Stat::IsConstant, true.into()),
                    ])));
                }
                MinMaxResult::MinMax(min, max) => {
                    return Ok(StatsSet::from(HashMap::from([
                        (Stat::Min, (*min).into()),
                        (Stat::Max, (*max).into()),
                        (Stat::IsConstant, false.into()),
                    ])));
                }
            }
        }

        if stat == Stat::IsConstant {
            let first = self[0];
            let is_constant = self.iter().all(|x| first.is_eq(*x));
            return Ok(StatsSet::from(HashMap::from([(
                Stat::IsConstant,
                is_constant.into(),
            )])));
        }

        if stat == Stat::NullCount {
            return Ok(StatsSet::from(HashMap::from([(
                Stat::NullCount,
                0u64.into(),
            )])));
        }

        if stat == Stat::IsSorted {
            let mut sorted = true;
            let mut prev = self[0];
            for next in self.iter().skip(1) {
                if matches!(next.compare(prev), Ordering::Less) {
                    sorted = false;
                    break;
                }
                prev = *next;
            }

            if sorted {
                return Ok(StatsSet::from(HashMap::from([(
                    Stat::IsSorted,
                    true.into(),
                )])));
            } else {
                return Ok(StatsSet::from(HashMap::from([
                    (Stat::IsSorted, false.into()),
                    (Stat::IsStrictSorted, false.into()),
                ])));
            }
        }

        if stat == Stat::IsStrictSorted {
            let mut strict_sorted = true;
            let mut prev = self[0];
            for next in self.iter().skip(1) {
                if !matches!(prev.compare(*next), Ordering::Less) {
                    strict_sorted = false;
                    break;
                }
                prev = *next;
            }

            if strict_sorted {
                return Ok(StatsSet::from(HashMap::from([
                    (Stat::IsSorted, true.into()),
                    (Stat::IsStrictSorted, true.into()),
                ])));
            } else {
                return Ok(StatsSet::from(HashMap::from([(
                    Stat::IsStrictSorted,
                    false.into(),
                )])));
            }
        }

        if stat == Stat::RunCount {
            let mut run_count = 0;
            let mut prev = self[0];
            for next in self.iter().skip(1) {
                if !prev.is_eq(*next) {
                    run_count += 1;
                    prev = *next;
                }
            }
            return Ok(StatsSet::from(HashMap::from([(
                Stat::RunCount,
                run_count.into(),
            )])));
        }

        if stat == Stat::BitWidthFreq || stat == Stat::TrailingZeroFreq {
            let mut stats = BitWidthAccumulator::new(self[0]);
            self.iter().skip(1).for_each(|next| stats.next(*next));
            return Ok(stats.finish());
        }

        Ok(StatsSet::new())
    }
}

struct NullableValues<'a, T: PStatsType>(&'a [T], &'a BooleanBuffer);

macro_rules! accumulate_stats {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        match $self {
            Stat::BitWidthFreq | Stat::TrailingZeroFreq => __with__! { BitWidthAccumulator },
            _ => __with__! { StatsAccumulator },
        }
    })
}

impl<T: PStatsType> ArrayStatisticsCompute for NullableValues<'_, T> {
    fn compute_statistics(&self, stat: Stat) -> VortexResult<StatsSet> {
        let values = self.0;
        if values.is_empty() || stat == Stat::TrueCount {
            return Ok(StatsSet::new());
        }

        if stat == Stat::NullCount {
            return Ok(StatsSet::from(HashMap::from([(
                Stat::NullCount,
                (values.len() - self.1.count_set_bits()).into(),
            )])));
        }

        let mut set_indices = self.1.set_indices();
        let Some(first_non_null) = set_indices.next() else {
            return Ok(StatsSet::nulls(
                values.len(),
                &DType::Primitive(T::PTYPE, Nullability::Nullable),
            ));
        };

        accumulate_stats!(stat, |$ACC| {
            let mut acc = $ACC::new(values[first_non_null]);
            acc.n_nulls(first_non_null);
            let last_non_null = set_indices.fold(first_non_null, |prev_set_bit, next| {
                let n_nulls = next - prev_set_bit - 1;
                acc.n_nulls(n_nulls);
                acc.next(values[next]);
                next
            });
            acc.n_nulls(values.len() - last_non_null - 1);
            return Ok(acc.finish());
        });
    }
}

trait BitWidth {
    fn bit_width(self) -> u32;
    fn trailing_zeros(self) -> u32;
}

macro_rules! int_bit_width {
    ($T:ty) => {
        impl BitWidth for $T {
            fn bit_width(self) -> u32 {
                Self::BITS - PrimInt::leading_zeros(self)
            }

            fn trailing_zeros(self) -> u32 {
                PrimInt::trailing_zeros(self)
            }
        }
    };
}

int_bit_width!(u8);
int_bit_width!(u16);
int_bit_width!(u32);
int_bit_width!(u64);
int_bit_width!(i8);
int_bit_width!(i16);
int_bit_width!(i32);
int_bit_width!(i64);

// TODO(ngates): just skip counting this in the implementation.
macro_rules! float_bit_width {
    ($T:ty) => {
        impl BitWidth for $T {
            fn bit_width(self) -> u32 {
                (size_of::<Self>() * 8) as u32
            }

            fn trailing_zeros(self) -> u32 {
                0
            }
        }
    };
}

float_bit_width!(f16);
float_bit_width!(f32);
float_bit_width!(f64);

struct BitWidthAccumulator<T: PStatsType> {
    bit_widths: Vec<u64>,
    trailing_zeros: Vec<u64>,
    _marker: PhantomData<T>,
}

impl<T: PStatsType> BitWidthAccumulator<T> {
    fn new(first_value: T) -> Self {
        let mut stats = Self {
            bit_widths: vec![0; size_of::<T>() * 8 + 1],
            trailing_zeros: vec![0; size_of::<T>() * 8 + 1],
            _marker: PhantomData,
        };
        stats.bit_widths[first_value.bit_width() as usize] += 1;
        stats.trailing_zeros[first_value.trailing_zeros() as usize] += 1;
        stats
    }

    fn n_nulls(&mut self, n_nulls: usize) {
        self.bit_widths[0] += n_nulls as u64;
        self.trailing_zeros[T::PTYPE.bit_width()] += n_nulls as u64;
    }

    pub fn next(&mut self, next: T) {
        self.bit_widths[next.bit_width() as usize] += 1;
        self.trailing_zeros[next.trailing_zeros() as usize] += 1;
    }

    pub fn finish(self) -> StatsSet {
        StatsSet::from(HashMap::from([
            (Stat::BitWidthFreq, self.bit_widths.into()),
            (Stat::TrailingZeroFreq, self.trailing_zeros.into()),
        ]))
    }
}

struct StatsAccumulator<T: PStatsType> {
    prev: T,
    min: T,
    max: T,
    is_sorted: bool,
    is_strict_sorted: bool,
    run_count: usize,
    null_count: usize,
    nan_count: usize,
    len: usize,
}

impl<T: PStatsType> StatsAccumulator<T> {
    fn new(first_value: T) -> Self {
        Self {
            prev: first_value,
            min: first_value,
            max: first_value,
            is_sorted: true,
            is_strict_sorted: true,
            run_count: 1,
            null_count: 0,
            nan_count: first_value.is_nan().then_some(1).unwrap_or_default(),
            len: 1,
        }
    }

    fn n_nulls(&mut self, n_nulls: usize) {
        self.null_count += n_nulls;
        self.len += n_nulls;
    }

    pub fn next(&mut self, next: T) {
        self.len += 1;

        if next.is_nan() {
            self.nan_count += 1;
        }

        if next.is_eq(self.prev) {
            self.is_strict_sorted = false;
        } else {
            if matches!(next.compare(self.prev), Ordering::Less) {
                self.is_sorted = false;
            }
            self.run_count += 1;
        }
        if matches!(next.compare(self.min), Ordering::Less) {
            self.min = next;
        } else if matches!(next.compare(self.max), Ordering::Greater) {
            self.max = next;
        }
        self.prev = next;
    }

    pub fn finish(self) -> StatsSet {
        let is_constant = (self.min == self.max && self.null_count == 0 && self.nan_count == 0)
            || self.null_count == self.len;

        StatsSet::from(HashMap::from([
            (Stat::Min, self.min.into()),
            (Stat::Max, self.max.into()),
            (Stat::NullCount, self.null_count.into()),
            (Stat::IsConstant, is_constant.into()),
            (Stat::IsSorted, self.is_sorted.into()),
            (
                Stat::IsStrictSorted,
                (self.is_sorted && self.is_strict_sorted).into(),
            ),
            (Stat::RunCount, self.run_count.into()),
        ]))
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

    use crate::array::primitive::PrimitiveArray;
    use crate::stats::{ArrayStatistics, Stat};

    #[test]
    fn stats() {
        let arr = PrimitiveArray::from(vec![1, 2, 3, 4, 5]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let is_sorted = arr.statistics().compute_is_sorted().unwrap();
        let is_strict_sorted = arr.statistics().compute_is_strict_sorted().unwrap();
        let is_constant = arr.statistics().compute_is_constant().unwrap();
        let bit_width_freq = arr.statistics().compute_bit_width_freq().unwrap();
        let trailing_zeros_freq = arr.statistics().compute_trailing_zero_freq().unwrap();
        let run_count = arr.statistics().compute_run_count().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
        assert!(is_sorted);
        assert!(is_strict_sorted);
        assert!(!is_constant);
        assert_eq!(
            bit_width_freq,
            vec![
                0usize, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(
            trailing_zeros_freq,
            vec![
                // 1, 3, 5 have 0 trailing zeros
                // 2 has 1 trailing zero, 4 has 2 trailing zeros
                3usize, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(run_count, 5);
    }

    #[test]
    fn stats_u8() {
        let arr = PrimitiveArray::from(vec![1u8, 2, 3, 4, 5]);
        let min: u8 = arr.statistics().compute_min().unwrap();
        let max: u8 = arr.statistics().compute_max().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 5);
    }

    #[test]
    fn nullable_stats_u8() {
        let arr = PrimitiveArray::from_nullable_vec(vec![None, None, Some(1i32), Some(2), None]);
        let min: i32 = arr.statistics().compute_min().unwrap();
        let max: i32 = arr.statistics().compute_max().unwrap();
        let null_count: usize = arr.statistics().compute_null_count().unwrap();
        let is_strict_sorted: bool = arr.statistics().compute_is_strict_sorted().unwrap();
        assert_eq!(min, 1);
        assert_eq!(max, 2);
        assert_eq!(null_count, 3);
        assert!(is_strict_sorted);
    }

    #[test]
    fn all_null() {
        let arr = PrimitiveArray::from_nullable_vec(vec![Option::<i32>::None, None, None]);
        let min: Option<Scalar> = arr.statistics().compute(Stat::Min);
        let max: Option<Scalar> = arr.statistics().compute(Stat::Max);
        let null_i32 = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
        assert_eq!(min, Some(null_i32.clone()));
        assert_eq!(max, Some(null_i32));
    }
}
