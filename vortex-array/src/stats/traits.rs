// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, vortex_panic};
use vortex_scalar::{Scalar, ScalarValue};

use super::{Precision, Stat, StatType};

/// Trait for providing statistical information about arrays.
///
/// This trait allows querying various statistics about an array's data,
/// such as minimum, maximum, null count, etc.
pub trait StatsProvider {
    /// Get the value of a specific statistic.
    ///
    /// Returns `None` if the statistic is not available or not computed.
    fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>>;

    /// Count of stored stats with known values.
    fn len(&self) -> usize;

    /// Predicate equivalent to a [len][Self::len] of zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S> StatsProviderExt for S where S: StatsProvider {}

/// Extension trait providing additional convenience methods for statistics.
///
/// This trait is automatically implemented for all types that implement [`StatsProvider`].
pub trait StatsProviderExt: StatsProvider {
    /// Get a statistic as a typed scalar value.
    ///
    /// This converts the raw scalar value to the appropriate type based on the data type.
    fn get_scalar(&self, stat: Stat, dtype: &DType) -> Option<Precision<Scalar>> {
        let stat_dtype = stat
            .dtype(dtype)
            .vortex_expect("getting scalar for stat dtype");
        self.get(stat).map(|v| v.into_scalar(stat_dtype))
    }

    /// Get a statistic as a bounded value for a specific stat type.
    ///
    /// This is useful for stats that have upper/lower bounds like min/max.
    fn get_scalar_bound<S: StatType<Scalar>>(&self, dtype: &DType) -> Option<S::Bound> {
        self.get_scalar(S::STAT, dtype).map(|v| v.bound::<S>())
    }

    /// Get a statistic as a specific type.
    ///
    /// This method attempts to convert the scalar value to the requested type.
    fn get_as<T: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<Precision<T>> {
        self.get(stat).map(|v| {
            v.map(|v| {
                T::try_from(&v).unwrap_or_else(|err| {
                    vortex_panic!(
                        err,
                        "Failed to get stat {} as {}",
                        stat,
                        std::any::type_name::<T>()
                    )
                })
            })
        })
    }

    /// Get a statistic as a bound for a specific stat type and value type.
    fn get_as_bound<S, U>(&self) -> Option<S::Bound>
    where
        S: StatType<U>,
        U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>,
    {
        self.get_as::<U>(S::STAT).map(|v| v.bound::<S>())
    }
}
