// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::ops::Deref;

use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;

/// Default alignment for device-to-host buffer copies.
pub const ALIGNMENT_TO_HOST_COPY: Alignment = Alignment::new(256);

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

    /// Create a new 1-byte alignment.
    #[inline]
    pub const fn none() -> Self {
        Self::new(1)
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

    /// The largest valid alignment: the greatest power of 2 that fits into a `u16`.
    pub const MAX: Alignment = Alignment::new(1 << 15);

    /// Check if `self` alignment is a "larger" than `other` alignment.
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
    pub const fn is_aligned_to(&self, other: Alignment) -> bool {
        // Since both alignments are powers of 2, divisibility is equivalent to ordering.
        self.0 >= other.0
    }

    /// Check if the given byte offset (or length) is a multiple of this alignment.
    ///
    /// ## Example
    ///
    /// ```
    /// use vortex_buffer::Alignment;
    ///
    /// let a = Alignment::new(4);
    /// assert!(a.is_offset_aligned(8));
    /// assert!(!a.is_offset_aligned(2));
    /// ```
    #[inline]
    pub const fn is_offset_aligned(&self, offset: usize) -> bool {
        // Alignment is always a power of 2, so a mask test is equivalent to `offset % self == 0`.
        offset & (self.0 - 1) == 0
    }

    /// Check if the given pointer is aligned to this alignment.
    #[inline]
    pub fn is_ptr_aligned<T>(&self, ptr: *const T) -> bool {
        self.is_offset_aligned(ptr.addr())
    }

    /// Returns the log2 of the alignment.
    pub fn exponent(&self) -> u8 {
        u8::try_from(self.0.trailing_zeros())
            .vortex_expect("alignment fits into u16, so exponent fits in u7")
    }

    /// Create from the log2 exponent of the alignment.
    ///
    /// ## Panics
    ///
    /// Panics if `alignment` is not a power of 2, or is greater than `u16::MAX`.
    #[inline]
    pub const fn from_exponent(exponent: u8) -> Self {
        Self::new(1 << exponent)
    }
}

impl Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Deref for Alignment {
    type Target = usize;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<usize> for Alignment {
    #[inline]
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl From<u16> for Alignment {
    #[inline]
    fn from(value: u16) -> Self {
        Self::new(usize::from(value))
    }
}

impl From<Alignment> for usize {
    #[inline]
    fn from(value: Alignment) -> Self {
        value.0
    }
}

impl From<Alignment> for u16 {
    #[inline]
    fn from(value: Alignment) -> Self {
        u16::try_from(value.0).vortex_expect("Alignment must fit into u16")
    }
}

impl From<Alignment> for u32 {
    #[inline]
    fn from(value: Alignment) -> Self {
        u32::try_from(value.0).vortex_expect("Alignment must fit into u32")
    }
}

impl TryFrom<u32> for Alignment {
    type Error = VortexError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let value = usize::try_from(value)
            .map_err(|_| vortex_err!("Alignment must fit into usize, got {value}"))?;

        if value == 0 {
            return Err(vortex_err!("Alignment must be greater than 0"));
        }
        if value > u16::MAX as usize {
            return Err(vortex_err!("Alignment must fit into u16, got {value}"));
        }
        if !value.is_power_of_two() {
            return Err(vortex_err!("Alignment must be a power of 2, got {value}"));
        }

        Ok(Self(value))
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
    fn alignment_exponent() {
        let alignment = Alignment::new(1024);
        assert_eq!(alignment.exponent(), 10);
        assert_eq!(Alignment::from_exponent(10), alignment);
    }

    #[test]
    fn is_aligned_to() {
        assert!(Alignment::new(1).is_aligned_to(Alignment::new(1)));
        assert!(Alignment::new(2).is_aligned_to(Alignment::new(1)));
        assert!(Alignment::new(4).is_aligned_to(Alignment::new(1)));
        assert!(!Alignment::new(1).is_aligned_to(Alignment::new(2)));
    }

    #[test]
    fn try_from_u32() {
        match Alignment::try_from(8u32) {
            Ok(alignment) => assert_eq!(alignment, Alignment::new(8)),
            Err(err) => panic!("unexpected error for valid alignment: {err}"),
        }
        assert!(Alignment::try_from(0u32).is_err());
        assert!(Alignment::try_from(3u32).is_err());
    }

    #[test]
    fn into_u32() {
        let alignment = Alignment::new(64);
        assert_eq!(u32::from(alignment), 64u32);
    }
}
