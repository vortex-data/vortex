use std::fmt::Debug;

use datafusion_common::stats::Precision as DFPrecision;

use crate::stats::Precision;

mod private {
    use crate::stats::Precision;

    pub trait Sealed {}
    impl<T> Sealed for Precision<T> where T: std::fmt::Debug + Clone + PartialEq + Eq + PartialOrd {}
    impl<T> Sealed for Option<Precision<T>> where
        T: std::fmt::Debug + Clone + PartialEq + Eq + PartialOrd
    {
    }
}
pub trait PrecisionExt<T>: private::Sealed
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    /// Convert `Precision` to the datafusion equivalent.
    fn to_df(self) -> DFPrecision<T>;
}

impl<T> PrecisionExt<T> for Precision<T>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Precision::Exact(v) => DFPrecision::Exact(v),
            Precision::Inexact(v) => DFPrecision::Inexact(v),
        }
    }
}

impl<T> PrecisionExt<T> for Option<Precision<T>>
where
    T: Debug + Clone + PartialEq + Eq + PartialOrd,
{
    fn to_df(self) -> DFPrecision<T> {
        match self {
            Some(v) => v.to_df(),
            None => DFPrecision::Absent,
        }
    }
}
