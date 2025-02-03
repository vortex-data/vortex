use std::cmp::Ordering;

pub trait PartialMin: PartialOrd<Self>
where
    Self: Sized,
{
    /// Returns the minimum of two values, if they are comparable.
    #[inline]
    fn partial_min(self, other: Self) -> Option<Self> {
        if self.partial_cmp(&other)? == Ordering::Less {
            Some(self)
        } else {
            Some(other)
        }
    }
}

pub fn partial_min<T: PartialOrd + PartialMin>(a: T, b: T) -> Option<T> {
    a.partial_min(b)
}

impl<T: PartialOrd> PartialMin for T {}

pub trait PartialMax: PartialOrd<Self>
where
    Self: Sized,
{
    /// Returns the maximum of two values, if they are comparable.
    #[inline]
    fn partial_max(self, other: Self) -> Option<Self> {
        if self.partial_cmp(&other)? == Ordering::Greater {
            Some(self)
        } else {
            Some(other)
        }
    }
}

impl<T: PartialOrd> PartialMax for T {}

pub fn partial_max<T: PartialOrd + PartialMax>(a: T, b: T) -> Option<T> {
    a.partial_max(b)
}
