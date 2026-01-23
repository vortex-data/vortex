// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use itertools::Itertools;
use num_traits::AsPrimitive;
use static_assertions::assert_eq_align;
use static_assertions::assert_eq_size;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_dtype::NativePType;
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
    /// Creates a new inlined representation from the provided value of constant size.
    #[inline]
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

impl BinaryView {
    /// Maximum size of an inlined binary value.
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
                _ref: Ref {
                    size: u32::try_from(value.len()).vortex_expect("value length must fit in u32"),
                    prefix: value[0..4]
                        .try_into()
                        .vortex_expect("prefix must be exactly 4 bytes"),
                    buffer_index: block,
                    offset,
                },
            },
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

/// Convert an offsets buffer to a buffer of element lengths.
#[inline]
pub fn offsets_to_lengths<P: NativePType>(offsets: &[P]) -> Buffer<P> {
    offsets
        .iter()
        .tuple_windows::<(_, _)>()
        .map(|(&start, &end)| end - start)
        .collect()
}

/// Maximum number of buffer bytes that can be referenced by a single `BinaryView`
pub const MAX_BUFFER_LEN: usize = i32::MAX as usize;

/// Split a large buffer of input `bytes` holding string data
pub fn build_views<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    max_buffer_len: usize,
    mut bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::with_capacity(lens.len());

    let mut buffers = Vec::new();
    let mut buf_index = start_buf_index;

    let mut offset = 0;
    for &len in lens {
        let len = len.as_();
        assert!(len <= max_buffer_len, "values cannot exceed max_buffer_len");

        if (offset + len) > max_buffer_len {
            // Roll the buffer every 2GiB, to avoid overflowing VarBinView offset field
            let rest = bytes.split_off(offset);

            buffers.push(bytes.freeze());
            buf_index += 1;
            offset = 0;

            bytes = rest;
        }
        let view = BinaryView::make_view(&bytes[offset..][..len], buf_index, offset.as_());
        // SAFETY: we reserved the right capacity beforehand
        unsafe { views.push_unchecked(view) };
        offset += len;
    }

    if !bytes.is_empty() {
        buffers.push(bytes.freeze());
    }

    (buffers, views.freeze())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;

    use super::BinaryView;
    use crate::arrays::build_views::build_views;

    #[test]
    fn test_to_canonical_large() {
        // We are testing generating views for raw data that should look like
        //
        //    aaaaaaaaaaaaa ("a"*13)
        //    bbbbbbbbbbbbb ("b"*13)
        //    ccccccccccccc ("c"*13)
        //    ddddddddddddd ("d"*13)
        //
        // In real code, this would all fit in one buffer, but to unit test the splitting logic
        // we split buffers at length 26, which should result in two buffers for the output array.
        let raw_data =
            ByteBufferMut::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbbcccccccccccccddddddddddddd");
        let lens = vec![13u8; 4];

        let (buffers, views) = build_views(0, 26, raw_data, &lens);

        assert_eq!(
            buffers,
            vec![
                ByteBuffer::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbb"),
                ByteBuffer::copy_from("cccccccccccccddddddddddddd"),
            ]
        );

        assert_eq!(
            views.as_slice(),
            &[
                BinaryView::make_view(b"aaaaaaaaaaaaa", 0, 0),
                BinaryView::make_view(b"bbbbbbbbbbbbb", 0, 13),
                BinaryView::make_view(b"ccccccccccccc", 1, 0),
                BinaryView::make_view(b"ddddddddddddd", 1, 13),
            ]
        )
    }
}
