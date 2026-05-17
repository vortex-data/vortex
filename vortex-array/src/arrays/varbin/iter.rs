// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impls for [`VarBinArray`].
//!
//! Two entry points are offered:
//!
//! * [`IterArray::iter`] returns a polymorphic [`VarBinIter`] that handles
//!   any offset ptype. Offsets are normalized to `Buffer<usize>` once at
//!   construction (a single tight cast loop, ~30 µs / 100k elements on a
//!   modern desktop). The per-element path is then a monomorphic slice
//!   load — no per-tick ptype dispatch.
//!
//! * [`VarBinArrayExt::iter_offsets`] returns a [`VarBinTypedIter<O>`]
//!   parameterized over the offset type. Use it when you already know
//!   the offset ptype (e.g. inside `match_each_integer_ptype!`) — there
//!   is *no* upfront conversion and the per-tick cost is one `O` read +
//!   one `O → usize` cast (free for `u64` on 64-bit, a zero-extension
//!   move for narrower unsigned types, a sign-extension for signed).
//!
//! Both honor the contract that all decode work (offsets, validity)
//! happens once in the entry point, not per tick.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::VarBinArray;
use crate::arrays::primitive::array::iter::ValidityCursor;
use crate::arrays::varbin::VarBinArrayExt;
use crate::dtype::NativePType;
use crate::iter_array::IterArray;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Trait for offset values readable by the typed iterator. All of
/// `i8..i64` and `u8..u64` qualify; values are widened to `usize`.
pub trait VarBinOffset: NativePType + Copy {
    /// Widen this offset to a `usize` byte index.
    fn to_byte_index(self) -> usize;
}

macro_rules! impl_offset {
    ($($t:ty),*) => {
        $(
            impl VarBinOffset for $t {
                #[inline(always)]
                fn to_byte_index(self) -> usize {
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    { self as usize }
                }
            }
        )*
    };
}
impl_offset!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Polymorphic iterator over a [`VarBinArray`]'s byte slices.
///
/// Returned by [`IterArray::iter`]. Internally specializes on the common
/// offset ptype (`u32`) so the common path is zero-overhead: refcount-
/// clone of the existing `Buffer<u32>`, per-tick u32 read + cast to
/// usize. Rare ptypes (i8/u8/i16/u16/i32/i64/u64) are normalized to
/// `Buffer<usize>` once at construction.
pub struct VarBinIter<'a> {
    inner: VarBinIterInner,
    bytes: &'a [u8],
    pos: usize,
    len: usize,
    validity: ValidityState,
}

enum VarBinIterInner {
    /// Fast path: refcount-clone of an existing `Buffer<u32>`.
    U32 { _buf: Buffer<u32>, ptr: *const u32 },
    /// Fallback: offsets normalized to `Buffer<usize>` once.
    Usize { offsets: Buffer<usize> },
}

// SAFETY: the raw pointer is owned via the adjacent `_buf`.
unsafe impl Send for VarBinIterInner {}

/// Typed iterator over a [`VarBinArray`]'s byte slices.
///
/// Returned by [`VarBinArrayExt::iter_offsets`]. No upfront offset
/// conversion; per tick reads an `O` and widens to `usize`.
pub struct VarBinTypedIter<'a, O: VarBinOffset> {
    _offsets_owner: Buffer<O>,
    offsets_ptr: *const O,
    bytes: &'a [u8],
    pos: usize,
    len: usize,
    validity: ValidityState,
}

// SAFETY: the raw pointer is owned via the adjacent `_offsets_owner`.
unsafe impl<O: VarBinOffset + Send> Send for VarBinTypedIter<'_, O> {}

enum ValidityState {
    AllValid,
    AllInvalid,
    WithValidity(ValidityCursor),
}

impl<'a> Iterator for VarBinIter<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        // The variant never changes during iteration, so the branch
        // predictor pins on one arm and the body inlines.
        let (start, end) = match &self.inner {
            VarBinIterInner::U32 { ptr, .. } => unsafe {
                (*ptr.add(self.pos) as usize, *ptr.add(self.pos + 1) as usize)
            },
            VarBinIterInner::Usize { offsets } => {
                let s = offsets.as_slice();
                (s[self.pos], s[self.pos + 1])
            }
        };
        self.pos += 1;
        let slice = &self.bytes[start..end];
        match &mut self.validity {
            ValidityState::AllValid => Some(Some(slice)),
            ValidityState::AllInvalid => Some(None),
            ValidityState::WithValidity(v) => {
                let valid = v.next_bit().unwrap_or(false);
                Some(valid.then_some(slice))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.pos;
        (n, Some(n))
    }
}

impl ExactSizeIterator for VarBinIter<'_> {}

impl<'a, O: VarBinOffset> Iterator for VarBinTypedIter<'a, O> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        // SAFETY: pos < len, and the offsets buffer has length len + 1
        // by the VarBin invariant.
        let (start, end) = unsafe {
            (
                (*self.offsets_ptr.add(self.pos)).to_byte_index(),
                (*self.offsets_ptr.add(self.pos + 1)).to_byte_index(),
            )
        };
        self.pos += 1;
        let slice = &self.bytes[start..end];
        match &mut self.validity {
            ValidityState::AllValid => Some(Some(slice)),
            ValidityState::AllInvalid => Some(None),
            ValidityState::WithValidity(v) => {
                let valid = v.next_bit().unwrap_or(false);
                Some(valid.then_some(slice))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.pos;
        (n, Some(n))
    }
}

impl<O: VarBinOffset> ExactSizeIterator for VarBinTypedIter<'_, O> {}

fn make_validity(v: Validity, len: usize) -> ValidityState {
    match v {
        Validity::NonNullable | Validity::AllValid => ValidityState::AllValid,
        Validity::AllInvalid => {
            // Length is recorded on the iterator itself; the AllInvalid
            // variant just signals to yield None for every position.
            let _ = len;
            ValidityState::AllInvalid
        }
        Validity::Array(v) => {
            #[expect(deprecated)]
            let bits: BitBuffer = v.to_bool().into_bit_buffer();
            ValidityState::WithValidity(ValidityCursor::new(bits))
        }
    }
}

impl IterArray<[u8]> for VarBinArray {
    type Iter<'a> = VarBinIter<'a>;

    fn iter(&self) -> Self::Iter<'_> {
        #[expect(deprecated)]
        let offsets_arr = self.offsets().to_primitive();
        let bytes: &[u8] = self.bytes().as_slice();
        let len = self.len();
        let inner = if offsets_arr.ptype() == crate::dtype::PType::U32 {
            // Fast path: refcount-clone of the existing Buffer<u32>.
            let buf = offsets_arr.into_buffer::<u32>();
            let ptr = buf.as_slice().as_ptr();
            VarBinIterInner::U32 { _buf: buf, ptr }
        } else {
            // Fallback: normalize to Buffer<usize> once.
            let offsets: Buffer<usize> = match_each_integer_ptype!(offsets_arr.ptype(), |O| {
                let src = offsets_arr.as_slice::<O>();
                Buffer::<usize>::from_iter(src.iter().map(|v| (*v).to_byte_index()))
            });
            VarBinIterInner::Usize { offsets }
        };
        let validity = self
            .validity()
            .vortex_expect("varbin validity should be derivable");
        VarBinIter {
            inner,
            bytes,
            pos: 0,
            len,
            validity: make_validity(validity, len),
        }
    }
}

/// Construct a typed varbin iterator with no upfront offset conversion.
///
/// `O` must match the array's offsets ptype. Panics otherwise.
pub fn iter_offsets<O: VarBinOffset>(arr: &VarBinArray) -> VarBinTypedIter<'_, O> {
    #[expect(deprecated)]
    let offsets_arr = arr.offsets().to_primitive();
    assert_eq!(
        offsets_arr.ptype(),
        O::PTYPE,
        "VarBinArray::iter_offsets called with O = {} but offsets ptype is {}",
        O::PTYPE,
        offsets_arr.ptype()
    );
    let bytes: &[u8] = arr.bytes().as_slice();
    let len = arr.len();
    let buf: Buffer<O> = offsets_arr.into_buffer::<O>();
    let offsets_ptr = buf.as_slice().as_ptr();
    let validity = arr
        .validity()
        .vortex_expect("varbin validity should be derivable");
    VarBinTypedIter {
        _offsets_owner: buf,
        offsets_ptr,
        bytes,
        pos: 0,
        len,
        validity: make_validity(validity, len),
    }
}
