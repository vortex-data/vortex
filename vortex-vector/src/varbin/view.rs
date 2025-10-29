// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use static_assertions::{assert_eq_align, assert_eq_size};
use vortex_error::VortexUnwrap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct Inlined {
    pub(super) size: u32,
    pub(super) data: [u8; BinaryView::MAX_INLINED_SIZE],
}

impl Inlined {
    fn new<const N: usize>(value: &[u8]) -> Self {
        let mut inlined = Self {
            size: N.try_into().vortex_unwrap(),
            data: [0u8; BinaryView::MAX_INLINED_SIZE],
        };
        inlined.data[..N].copy_from_slice(&value[..N]);
        inlined
    }

    #[inline]
    pub fn value(&self) -> &[u8] {
        &self.data[0..(self.size as usize)]
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
pub struct Ref {
    pub(super) size: u32,
    pub(super) prefix: [u8; 4],
    pub(super) buffer_index: u32,
    pub(super) offset: u32,
}

impl Ref {
    pub fn new(size: u32, prefix: [u8; 4], buffer_index: u32, offset: u32) -> Self {
        Self {
            size,
            prefix,
            buffer_index,
            offset,
        }
    }

    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    #[inline]
    pub fn buffer_index(&self) -> u32 {
        self.buffer_index
    }

    #[inline]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    #[inline]
    pub fn prefix(&self) -> &[u8; 4] {
        &self.prefix
    }

    #[inline]
    pub fn as_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }

    #[inline]
    pub fn with_buffer_and_offset(&self, buffer_index: u32, offset: u32) -> Ref {
        Self::new(self.size, self.prefix, buffer_index, offset)
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub union BinaryView {
    // Numeric representation. This is logically `u128`, but we split it into the high and low
    // bits to preserve the alignment.
    pub(super) le_bytes: [u8; 16],

    // Inlined representation: strings <= 12 bytes
    pub(super) inlined: Inlined,

    // Reference type: strings > 12 bytes.
    pub(super) _ref: Ref,
}

assert_eq_size!(BinaryView, [u8; 16]);
assert_eq_size!(Inlined, [u8; 16]);
assert_eq_size!(Ref, [u8; 16]);
assert_eq_align!(BinaryView, u128);

impl PartialEq for BinaryView {
    fn eq(&self, other: &Self) -> bool {
        let a = unsafe { std::mem::transmute::<&BinaryView, &[u8; 16]>(self) };
        let b = unsafe { std::mem::transmute::<&BinaryView, &[u8; 16]>(other) };
        a == b
    }
}
impl Eq for BinaryView {}

impl Hash for BinaryView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { std::mem::transmute::<&BinaryView, &[u8; 16]>(self) }.hash(state);
    }
}

impl Default for BinaryView {
    fn default() -> Self {
        Self::make_view(&[], 0, 0)
    }
}

impl BinaryView {
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
            _ => Self {
                _ref: Ref::new(
                    u32::try_from(value.len()).vortex_unwrap(),
                    value[0..4].try_into().vortex_unwrap(),
                    block,
                    offset,
                ),
            },
        }
    }

    /// Create a new empty view
    #[inline]
    pub fn empty_view() -> Self {
        Self::new_inlined(&[])
    }

    /// Create a new inlined binary view
    #[inline]
    pub fn new_inlined(value: &[u8]) -> Self {
        assert!(
            value.len() <= Self::MAX_INLINED_SIZE,
            "expected inlined value to be <= 12 bytes, was {}",
            value.len()
        );

        Self::make_view(value, 0, 0)
    }

    #[inline]
    pub fn len(&self) -> u32 {
        unsafe { self.inlined.size }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn is_inlined(&self) -> bool {
        self.len() <= (Self::MAX_INLINED_SIZE as u32)
    }

    pub fn as_inlined(&self) -> &Inlined {
        unsafe { &self.inlined }
    }

    pub fn as_view(&self) -> &Ref {
        unsafe { &self._ref }
    }

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
