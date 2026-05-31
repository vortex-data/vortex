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
    pub const MAX_INLINED_SIZE: usize = 12;

    /// Create a view from a value, block and offset
    ///
    /// Depending on the length of the provided value either a new inlined
    /// or a reference view will be constructed.
    ///
    /// The inline payload is filled by [`copy_inline`] — a handful of overlapping
    /// fixed-width copies. The previous implementation matched on all 13 inline
    /// lengths, which (on x86) compiled to a jump-table indirect branch that
    /// mispredicts on every length change — costly on this per-row hot path where
    /// short strings dominate. `copy_inline` emits no jump table and (unlike a
    /// plain variable-length `copy_from_slice`) no `memcpy` call: verified with
    /// `--emit asm` — the inline fill drops from 108 instructions with an indirect
    /// jump to ~40 with only well-predicted length compares.
    #[inline(never)]
    pub fn make_view(value: &[u8], block: u32, offset: u32) -> Self {
        let len = value.len();
        if len <= Self::MAX_INLINED_SIZE {
            let mut inlined = Inlined {
                size: len as u32,
                data: [0u8; Self::MAX_INLINED_SIZE],
            };
            copy_inline(&mut inlined.data, value);
            Self { inlined }
        } else {
            Self {
                _ref: Ref {
                    size: u32::try_from(len).vortex_expect("value length must fit in u32"),
                    prefix: value[0..4]
                        .try_into()
                        .ok()
                        .vortex_expect("prefix must be exactly 4 bytes"),
                    buffer_index: block,
                    offset,
                },
            }
        }
    }

    /// Create a new empty view
    #[inline]
    pub fn empty_view() -> Self {
        Self { le_bytes: [0; 16] }
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

    /// Returns the binary view as u128 representation.
    pub fn as_u128(&self) -> u128 {
        // SAFETY: binary view always safe to read as u128 LE bytes
        unsafe { u128::from_le_bytes(self.le_bytes) }
    }
}

/// Copy `src` (length `≤ 12`) into the front of `dst` using a few overlapping
/// fixed-width loads/stores rather than a variable-length `memcpy`.
///
/// For each size class the two copies overlap when the length isn't an exact
/// multiple of the width, which writes every byte exactly into place with no
/// per-byte loop and no jump table. `dst` is pre-zeroed by the caller, so the
/// `len == 0` case is a no-op.
#[inline(always)]
fn copy_inline(dst: &mut [u8; BinaryView::MAX_INLINED_SIZE], src: &[u8]) {
    let len = src.len();
    debug_assert!(len <= BinaryView::MAX_INLINED_SIZE);
    if len >= 8 {
        dst[0..8].copy_from_slice(&src[0..8]);
        dst[len - 8..len].copy_from_slice(&src[len - 8..len]);
    } else if len >= 4 {
        dst[0..4].copy_from_slice(&src[0..4]);
        dst[len - 4..len].copy_from_slice(&src[len - 4..len]);
    } else if len >= 2 {
        dst[0..2].copy_from_slice(&src[0..2]);
        dst[len - 2..len].copy_from_slice(&src[len - 2..len]);
    } else if len == 1 {
        dst[0] = src[0];
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
