use std::fmt::Display;
use std::ops::Deref;

use vortex_error::VortexExpect;

/// The alignment of a buffer.
///
/// This type is a wrapper around `usize` that ensures the alignment is a power of 2 and fits into
/// a `u16`.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Alignment(usize);

impl Alignment {
    /// Create a new alignment.
    ///
    /// ## Panics
    ///
    /// Panics if `align` is not a power of 2, or is greater than `u16::MAX`.
    #[inline]
    pub const fn new(align: usize) -> Self {
        assert!(align > 0, "Alignment must be greater than 0");
        assert!(align <= u16::MAX as usize, "Alignment must fit into u16");
        assert!(align.is_power_of_two(), "Alignment must be a power of 2");
        Self(align)
    }

    /// Create an alignment from the alignment of a type `T`.
    ///
    /// ## Example
    ///
    /// ```
    /// use vortex_buffer::Alignment;
    ///
    /// assert_eq!(Alignment::new(4), Alignment::of::<i32>());
    /// assert_eq!(Alignment::new(8), Alignment::of::<i64>());
    /// assert_eq!(Alignment::new(16), Alignment::of::<u128>());
    /// ```
    #[inline]
    pub const fn of<T>() -> Self {
        Self::new(align_of::<T>())
    }

    /// Check if this alignment is a "larger" than another alignment.
    ///
    /// ## Example
    ///
    /// ```
    /// use vortex_buffer::Alignment;
    ///
    /// let a = Alignment::new(4);
    /// let b = Alignment::new(2);
    /// assert!(a.is_aligned_to(b));
    /// assert!(!b.is_aligned_to(a));
    /// ```
    #[inline]
    pub fn is_aligned_to(&self, other: Alignment) -> bool {
        // Since we know alignments are powers of 2, we can compare them by checking if the number
        // of trailing zeros in the binary representation of the alignment is greater or equal.
        self.0.trailing_zeros() >= other.0.trailing_zeros()
    }
}

impl Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for Alignment {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<usize> for Alignment {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl From<u16> for Alignment {
    fn from(value: u16) -> Self {
        Self::new(usize::from(value))
    }
}

impl From<Alignment> for usize {
    fn from(value: Alignment) -> Self {
        value.0
    }
}

impl From<Alignment> for u16 {
    fn from(value: Alignment) -> Self {
        u16::try_from(value.0).vortex_expect("Alignment must fit into u16")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic]
    fn alignment_zero() {
        Alignment::new(0);
    }

    #[test]
    #[should_panic]
    fn alignment_overflow() {
        Alignment::new(u16::MAX as usize + 1);
    }

    #[test]
    #[should_panic]
    fn alignment_not_power_of_two() {
        Alignment::new(3);
    }

    #[test]
    fn is_aligned_to() {
        assert!(Alignment::new(1).is_aligned_to(Alignment::new(1)));
        assert!(Alignment::new(2).is_aligned_to(Alignment::new(1)));
        assert!(Alignment::new(4).is_aligned_to(Alignment::new(1)));
        assert!(!Alignment::new(1).is_aligned_to(Alignment::new(2)));
    }
}
