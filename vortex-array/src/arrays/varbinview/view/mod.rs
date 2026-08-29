// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The 16-byte view struct stored in variable-length binary vectors.

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use static_assertions::assert_eq_align;
use static_assertions::assert_eq_size;
use vortex_error::VortexExpect;

/// A view over a variable-length binary value.
///
/// Either an inlined representation (for values <= 12 bytes) or a reference
/// to an external buffer (for values > 12 bytes).
///
/// # Comparison words
///
/// Every view — inlined or reference — carries the value's length in its first four bytes and
/// the value's first four bytes, zero-padded, in its next four. Reading those fields answers most
/// comparisons from the view alone, which for a reference view saves a second, scattered read
/// into a data buffer. Those bytes are exposed as [`prefix`](Self::prefix) and packed into three
/// comparison words, each with a matching `_of` constructor for a value that is not in a view (a
/// scalar being compared against, say):
///
/// - [`head`](Self::head) pairs the length with the prefix. Equal values have equal heads, so an
///   unequal head rules out equality without reading either value.
/// - [`order_prefix`](Self::order_prefix) is the prefix as a big-endian integer, which orders the
///   same way the values it came from do.
/// - [`order_tail`](Self::order_tail) continues that order through the rest of an inlined value,
///   whose twelve bytes it and `order_prefix` cover between them.
///
/// Zero padding is what makes the ordering words order *refinements* rather than approximations:
/// if `a`'s word is below `b`'s then either they first differ at a byte both values have, or `a`
/// ran out of bytes first and is a proper prefix of `b`; either way `a < b`. Equal words say
/// nothing beyond the bytes they cover, so they fall through to the next word, to the values
/// themselves, or — once both words are exhausted on inlined values — to the lengths.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub union BinaryView {
    /// Numeric representation. This is logically `u128`, but we split it into the high and low
    /// bits to preserve the alignment.
    pub(crate) le_bytes: [u8; 16],

    /// Inlined representation: strings <= 12 bytes
    pub(crate) inlined: Inlined,

    /// Reference type: strings > 12 bytes.
    pub(crate) _ref: Ref,
}

assert_eq_align!(BinaryView, u128);
assert_eq_size!(BinaryView, [u8; 16]);
assert_eq_size!(Inlined, [u8; 16]);
assert_eq_size!(Ref, [u8; 16]);

/// Variant of a [`BinaryView`] that holds an inlined value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct Inlined {
    /// The size of the full value.
    pub size: u32,
    /// The full inlined value.
    pub data: [u8; BinaryView::MAX_INLINED_SIZE],
}

impl Inlined {
    /// Creates a new inlined representation from the provided value of constant size.
    fn new<const N: usize>(value: &[u8]) -> Self {
        debug_assert_eq!(value.len(), N);
        let mut inlined = Self {
            size: N.try_into().vortex_expect("inlined size must fit in u32"),
            data: [0u8; BinaryView::MAX_INLINED_SIZE],
        };
        inlined.data[..N].copy_from_slice(&value[..N]);
        inlined
    }

    /// Returns the full inlined value.
    #[inline]
    pub fn value(&self) -> &[u8] {
        &self.data[0..(self.size as usize)]
    }
}

/// Variant of a [`BinaryView`] that holds a reference to an external buffer.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
pub struct Ref {
    /// The size of the full value.
    pub size: u32,
    /// The prefix bytes of the value (first 4 bytes).
    pub prefix: [u8; 4],
    /// The index of the buffer where the full value is stored.
    pub buffer_index: u32,
    /// The offset within the buffer where the full value starts.
    pub offset: u32,
}

impl Ref {
    /// Returns the range within the buffer where the full value is stored.
    #[inline]
    pub fn as_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }

    /// Replaces the buffer index and offset of the reference, returning a new `Ref`.
    #[inline]
    pub fn with_buffer_and_offset(&self, buffer_index: u32, offset: u32) -> Ref {
        Self {
            size: self.size,
            prefix: self.prefix,
            buffer_index,
            offset,
        }
    }
}

impl BinaryView {
    /// Maximum size of an inlined binary value.
    ///
    /// cbindgen:ignore
    pub const MAX_INLINED_SIZE: usize = 12;

    /// Create a view from a value, block and offset
    ///
    /// Depending on the length of the provided value either a new inlined
    /// or a reference view will be constructed.
    ///
    /// Adapted from arrow-rs <https://github.com/apache/arrow-rs/blob/f4fde769ab6e1a9b75f890b7f8b47bc22800830b/arrow-array/src/builder/generic_bytes_view_builder.rs#L524>
    /// Explicitly enumerating inlined view produces code that avoids calling generic `ptr::copy_non_interleave` that's slower than explicit stores
    #[inline(never)]
    pub fn make_view(value: &[u8], block: u32, offset: u32) -> Self {
        match value.len() {
            0 => Self {
                inlined: Inlined::new::<0>(value),
            },
            1 => Self {
                inlined: Inlined::new::<1>(value),
            },
            2 => Self {
                inlined: Inlined::new::<2>(value),
            },
            3 => Self {
                inlined: Inlined::new::<3>(value),
            },
            4 => Self {
                inlined: Inlined::new::<4>(value),
            },
            5 => Self {
                inlined: Inlined::new::<5>(value),
            },
            6 => Self {
                inlined: Inlined::new::<6>(value),
            },
            7 => Self {
                inlined: Inlined::new::<7>(value),
            },
            8 => Self {
                inlined: Inlined::new::<8>(value),
            },
            9 => Self {
                inlined: Inlined::new::<9>(value),
            },
            10 => Self {
                inlined: Inlined::new::<10>(value),
            },
            11 => Self {
                inlined: Inlined::new::<11>(value),
            },
            12 => Self {
                inlined: Inlined::new::<12>(value),
            },
            _ => Self::new_ref(
                u32::try_from(value.len()).vortex_expect("value length must fit in u32"),
                value[0..4]
                    .try_into()
                    .ok()
                    .vortex_expect("prefix must be exactly 4 bytes"),
                block,
                offset,
            ),
        }
    }

    /// Create a new empty view
    #[inline]
    pub fn empty_view() -> Self {
        Self { le_bytes: [0; 16] }
    }

    /// Create a reference view directly from its components, without inspecting the value.
    ///
    /// `size` must be greater than [`MAX_INLINED_SIZE`], and `prefix` must hold the first four
    /// bytes of the value. This is the fast path for bulk view construction where the caller has
    /// already established that the value is too long to inline; it assembles the 16-byte view as a
    /// single `u128` so the compiler can emit one wide store per view.
    ///
    /// [`MAX_INLINED_SIZE`]: Self::MAX_INLINED_SIZE
    #[inline]
    pub fn new_ref(size: u32, prefix: [u8; 4], buffer_index: u32, offset: u32) -> Self {
        debug_assert!(size as usize > Self::MAX_INLINED_SIZE);
        // Matches the little-endian field order of `Ref` (size, prefix, buffer_index, offset),
        // consistent with `le_bytes` and the `From<u128>`/`as_u128` representation.
        Self::from(
            u128::from(size)
                | (u128::from(u32::from_le_bytes(prefix)) << 32)
                | (u128::from(buffer_index) << 64)
                | (u128::from(offset) << 96),
        )
    }

    /// Create a new inlined binary view
    ///
    /// # Panics
    ///
    /// Panics if the provided string is too long to inline.
    #[inline]
    pub fn new_inlined(value: &[u8]) -> Self {
        assert!(
            value.len() <= Self::MAX_INLINED_SIZE,
            "expected inlined value to be <= 12 bytes, was {}",
            value.len()
        );

        Self::make_view(value, 0, 0)
    }

    /// Returns the length of the binary value.
    #[inline]
    pub fn len(&self) -> u32 {
        unsafe { self.inlined.size }
    }

    /// Returns true if the binary value is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the binary value is inlined.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "MAX_INLINED_SIZE is a small constant"
    )]
    pub fn is_inlined(&self) -> bool {
        self.len() <= (Self::MAX_INLINED_SIZE as u32)
    }

    /// Returns the inlined representation of the binary value.
    pub fn as_inlined(&self) -> &Inlined {
        debug_assert!(self.is_inlined());
        unsafe { &self.inlined }
    }

    /// Returns the reference representation of the binary value.
    pub fn as_view(&self) -> &Ref {
        debug_assert!(!self.is_inlined());
        unsafe { &self._ref }
    }

    /// Returns a mutable reference to the reference representation of the binary value.
    pub fn as_view_mut(&mut self) -> &mut Ref {
        unsafe { &mut self._ref }
    }

    /// Returns the bytes of this value, reading out of `buffers` when it is not inlined.
    ///
    /// Unlike [`VarBinViewData::bytes_at`](crate::arrays::varbinview::VarBinViewData::bytes_at)
    /// this borrows
    /// from slices the caller has already resolved instead of cloning a buffer handle, so it is
    /// safe to call once per row in a loop.
    ///
    /// # Panics
    ///
    /// Panics if the view references a buffer or range not covered by `buffers`.
    #[inline]
    pub fn bytes<'a>(&'a self, buffers: &[&'a [u8]]) -> &'a [u8] {
        if self.is_inlined() {
            self.as_inlined().value()
        } else {
            let view = self.as_view();
            &buffers[view.buffer_index as usize][view.as_range()]
        }
    }

    /// The value's first four bytes, zero-padded for values shorter than four bytes.
    ///
    /// Both variants store them at the same offset: the leading bytes of an inlined value, or the
    /// `prefix` field of a reference view.
    #[inline]
    #[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
    pub fn prefix(&self) -> [u8; 4] {
        ((self.as_u128() >> 32) as u32).to_le_bytes()
    }

    /// The value's length paired with its [`prefix`](Self::prefix), as the view stores them.
    ///
    /// Equal values have equal heads, so an unequal head rules out equality without reading
    /// either value. See [the module docs](BinaryView#comparison-words).
    #[inline]
    #[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
    pub fn head(&self) -> u64 {
        self.as_u128() as u64
    }

    /// The value's [`prefix`](Self::prefix) as a big-endian `u32`, which orders the way the
    /// values do: a lower `order_prefix` means a lower value, and equal ones decide nothing.
    ///
    /// See [the module docs](BinaryView#comparison-words).
    #[inline]
    pub fn order_prefix(&self) -> u32 {
        u32::from_be_bytes(self.prefix())
    }

    /// Value bytes `4..12` of an inlined view as a big-endian `u64`, zero-padded past the value's
    /// length, continuing [`order_prefix`](Self::order_prefix) through the rest of the value.
    ///
    /// Meaningful only for an inlined view: a reference view stores its buffer index and offset in
    /// these bytes. Two inlined values are fully ordered by their order prefix, then their order
    /// tail, then their length. See [the module docs](BinaryView#comparison-words).
    #[inline]
    pub fn order_tail(&self) -> u64 {
        debug_assert!(self.is_inlined(), "order_tail of a reference view");
        ((self.as_u128() >> 64) as u64).swap_bytes()
    }

    /// The [`prefix`](Self::prefix) a view holding `value` would carry.
    #[inline]
    pub fn prefix_of(value: &[u8]) -> [u8; 4] {
        let mut prefix = [0u8; 4];
        let len = value.len().min(prefix.len());
        prefix[..len].copy_from_slice(&value[..len]);
        prefix
    }

    /// The [`head`](Self::head) a view holding `value` would carry.
    ///
    /// A value too long to live in a view (over `u32::MAX` bytes) saturates its length rather
    /// than wrapping it into the prefix bits.
    #[inline]
    pub fn head_of(value: &[u8]) -> u64 {
        let len = u32::try_from(value.len()).unwrap_or(u32::MAX);
        u64::from(len) | (u64::from(u32::from_le_bytes(Self::prefix_of(value))) << 32)
    }

    /// The [`order_prefix`](Self::order_prefix) a view holding `value` would carry.
    #[inline]
    pub fn order_prefix_of(value: &[u8]) -> u32 {
        u32::from_be_bytes(Self::prefix_of(value))
    }

    /// The [`order_tail`](Self::order_tail) an inlined view holding `value` would carry.
    ///
    /// Defined for a value of any length, but only comparable against a view's `order_tail` once
    /// that view is known to be inlined.
    #[inline]
    pub fn order_tail_of(value: &[u8]) -> u64 {
        let mut tail = [0u8; 8];
        let value = value.get(4..).unwrap_or_default();
        let len = value.len().min(tail.len());
        tail[..len].copy_from_slice(&value[..len]);
        u64::from_be_bytes(tail)
    }

    /// Returns the binary view as u128 representation.
    pub fn as_u128(&self) -> u128 {
        // SAFETY: binary view always safe to read as u128 LE bytes
        unsafe { u128::from_le_bytes(self.le_bytes) }
    }
}

impl From<u128> for BinaryView {
    fn from(value: u128) -> Self {
        BinaryView {
            le_bytes: value.to_le_bytes(),
        }
    }
}

impl From<Ref> for BinaryView {
    fn from(value: Ref) -> Self {
        BinaryView { _ref: value }
    }
}

impl PartialEq for BinaryView {
    fn eq(&self, other: &Self) -> bool {
        let a = unsafe { std::mem::transmute::<&BinaryView, &u128>(self) };
        let b = unsafe { std::mem::transmute::<&BinaryView, &u128>(other) };
        a == b
    }
}
impl Eq for BinaryView {}

impl Hash for BinaryView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { std::mem::transmute::<&BinaryView, &u128>(self) }.hash(state);
    }
}

impl Default for BinaryView {
    fn default() -> Self {
        Self::make_view(&[], 0, 0)
    }
}

impl fmt::Debug for BinaryView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("BinaryView");
        if self.is_inlined() {
            s.field("inline", &self.as_inlined());
        } else {
            s.field("ref", &self.as_view());
        }
        s.finish()
    }
}

#[cfg(test)]
mod tests;
