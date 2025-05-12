use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, vortex_panic};
use vortex_scalar::{Scalar, ScalarValue};

use super::{Precision, Stat, StatType};

pub trait StatsProvider {
    fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>>;

    /// Count of stored stats with known values.
    fn len(&self) -> usize;

    /// Predicate equivalent to a [len][Self::len] of zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S> StatsProviderExt for S where S: StatsProvider {}

pub trait StatsProviderExt: StatsProvider {
    fn get_scalar(&self, stat: Stat, dtype: &DType) -> Option<Precision<Scalar>> {
        let stat_dtype = stat
            .dtype(dtype)
            .vortex_expect("getting scalar for stat dtype");
        self.get(stat).map(|v| v.into_scalar(stat_dtype))
    }

    fn get_scalar_bound<S: StatType<Scalar>>(&self, dtype: &DType) -> Option<S::Bound> {
        self.get_scalar(S::STAT, dtype).map(|v| v.bound::<S>())
    }

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

    fn get_as_bound<S, U>(&self) -> Option<S::Bound>
    where
        S: StatType<U>,
        U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>,
    {
        self.get_as::<U>(S::STAT).map(|v| v.bound::<S>())
    }
}
