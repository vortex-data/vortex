// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexError;
use vortex_error::vortex_panic;
use vortex_scalar::Scalar;

use super::StatType;
use crate::expr::stats::Stat;
use crate::stats::Precision;

pub trait StatsProvider {
    fn get(&self, stat: Stat) -> Option<Precision<Scalar>>;

    /// Count of stored stats with known values.
    fn len(&self) -> usize;

    /// Predicate equivalent to a [len][Self::len] of zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S> StatsProviderExt for S where S: StatsProvider {}

pub trait StatsProviderExt: StatsProvider {
    fn get_scalar_bound<S: StatType<Scalar>>(&self) -> Option<S::Bound> {
        self.get(S::STAT).map(|v| v.bound::<S>())
    }

    fn get_as<T: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
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

    fn get_as_bound<S, U>(&self) -> Option<S::Bound>
    where
        S: StatType<U>,
        U: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        self.get_as::<U>(S::STAT).map(|v| v.bound::<S>())
    }
}
