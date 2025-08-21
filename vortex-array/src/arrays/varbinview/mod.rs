// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;

use static_assertions::{assert_eq_align, assert_eq_size};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{
    VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_ensure, vortex_err, vortex_panic,
};

use crate::builders::{ArrayBuilder, VarBinViewBuilder};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use crate::{Canonical, EncodingId, EncodingRef, vtable};

mod accessor;
mod compact;
mod compute;
mod ops;
mod serde;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct Inlined {
    size: u32,
    data: [u8; BinaryView::MAX_INLINED_SIZE],
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
    size: u32,
    prefix: [u8; 4],
    buffer_index: u32,
    offset: u32,
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
    pub fn to_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub union BinaryView {
    // Numeric representation. This is logically `u128`, but we split it into the high and low
    // bits to preserve the alignment.
    le_bytes: [u8; 16],

    // Inlined representation: strings <= 12 bytes
    inlined: Inlined,

    // Reference type: strings > 12 bytes.
    _ref: Ref,
}

assert_eq_size!(BinaryView, [u8; 16]);
assert_eq_size!(Inlined, [u8; 16]);
assert_eq_size!(Ref, [u8; 16]);
assert_eq_align!(BinaryView, u128);

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
        self.len() > 0
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

    /// Override the buffer reference with the given buffer_idx, only if this view is not inlined.
    #[inline(always)]
    pub fn with_buffer_idx(self, buffer_idx: u32) -> Self {
        if self.is_inlined() {
            self
        } else {
            // Referencing views must have their buffer_index adjusted with new offsets
            let view_ref = self.as_view();
            Self {
                _ref: Ref::new(
                    self.len(),
                    *view_ref.prefix(),
                    buffer_idx,
                    view_ref.offset(),
                ),
            }
        }
    }

    /// Shifts the buffer reference by the view by a given offset, useful when merging many
    /// varbinview arrays into one.
    #[inline(always)]
    pub fn offset_view(self, offset: u32) -> Self {
        if self.is_inlined() {
            self
        } else {
            // Referencing views must have their buffer_index adjusted with new offsets
            let view_ref = self.as_view();
            Self {
                _ref: Ref::new(
                    self.len(),
                    *view_ref.prefix(),
                    offset + view_ref.buffer_index(),
                    view_ref.offset(),
                ),
            }
        }
    }
}

impl From<u128> for BinaryView {
    fn from(value: u128) -> Self {
        BinaryView {
            le_bytes: value.to_le_bytes(),
        }
    }
}

impl Debug for BinaryView {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("BinaryView");
        if self.is_inlined() {
            s.field("inline", &"i".to_string());
        } else {
            s.field("ref", &"r".to_string());
        }
        s.finish()
    }
}

vtable!(VarBinView);

impl VTable for VarBinViewVTable {
    type Array = VarBinViewArray;
    type Encoding = VarBinViewEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.varbinview")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(VarBinViewEncoding.as_ref())
    }
}

/// A variable-length binary view array that stores strings and binary data efficiently.
///
/// This mirrors the Apache Arrow StringView/BinaryView array encoding and provides
/// an optimized representation for variable-length data with excellent performance
/// characteristics for both short and long strings.
///
/// ## Data Layout
///
/// The array uses a hybrid storage approach with two main components:
/// - **Views buffer**: Array of 16-byte `BinaryView` entries (one per logical element)
/// - **Data buffers**: Shared backing storage for strings longer than 12 bytes
///
/// ## View Structure
///
/// Commonly referred to as "German Strings", each 16-byte view entry contains either:
/// - **Inlined data**: For strings ≤ 12 bytes, the entire string is stored directly in the view
/// - **Reference data**: For strings > 12 bytes, contains:
///   - String length (4 bytes)
///   - First 4 bytes of string as prefix (4 bytes)
///   - Buffer index and offset (8 bytes total)
///
/// The following ASCII graphic is reproduced verbatim from the Arrow documentation:
///
/// ```text
///                         ┌──────┬────────────────────────┐
///                         │length│      string value      │
///    Strings (len <= 12)  │      │    (padded with 0)     │
///                         └──────┴────────────────────────┘
///                          0    31                      127
///
///                         ┌───────┬───────┬───────┬───────┐
///                         │length │prefix │  buf  │offset │
///    Strings (len > 12)   │       │       │ index │       │
///                         └───────┴───────┴───────┴───────┘
///                          0    31       63      95    127
/// ```
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::VarBinViewArray;
/// use vortex_dtype::{DType, Nullability};
/// use vortex_array::IntoArray;
///
/// // Create from an Iterator<Item = &str>
/// let array = VarBinViewArray::from_iter_str([
///         "inlined",
///         "this string is outlined"
/// ]);
///
/// assert_eq!(array.len(), 2);
///
/// // Access individual strings
/// let first = array.bytes_at(0);
/// assert_eq!(first.as_slice(), b"inlined"); // "short"
///
/// let second = array.bytes_at(1);
/// assert_eq!(second.as_slice(), b"this string is outlined"); // Long string
/// ```
#[derive(Clone, Debug)]
pub struct VarBinViewArray {
    dtype: DType,
    buffers: Arc<[ByteBuffer]>,
    views: Buffer<BinaryView>,
    validity: Validity,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct VarBinViewEncoding;

impl VarBinViewArray {
    fn validate(
        views: &Buffer<BinaryView>,
        buffers: &Arc<[ByteBuffer]>,
        dtype: &DType,
        validity: &Validity,
    ) -> VortexResult<()> {
        vortex_ensure!(
            validity.nullability() == dtype.nullability(),
            "validity {:?} incompatible with nullability {:?}",
            validity,
            dtype.nullability()
        );

        match dtype {
            DType::Utf8(_) => Self::validate_views(views, buffers, validity, |string| {
                std::str::from_utf8(string).is_ok()
            })?,
            DType::Binary(_) => Self::validate_views(views, buffers, validity, |_| true)?,
            _ => vortex_bail!("invalid DType {dtype}"),
        }

        Ok(())
    }

    fn validate_views<F>(
        views: &Buffer<BinaryView>,
        buffers: &Arc<[ByteBuffer]>,
        validity: &Validity,
        validator: F,
    ) -> VortexResult<()>
    where
        F: Fn(&[u8]) -> bool,
    {
        for (idx, &view) in views.iter().enumerate() {
            if validity.is_null(idx)? {
                continue;
            }

            if view.is_inlined() {
                // Validate the inline bytestring
                let bytes = &unsafe { view.inlined }.data[..view.len() as usize];
                vortex_ensure!(
                    validator(bytes),
                    "view at index {idx}: inlined bytes failed utf-8 validation"
                );
            } else {
                // Validate the view pointer
                let view = view.as_view();
                let buf_index = view.buffer_index as usize;
                let start_offset = view.offset as usize;
                let end_offset = start_offset.saturating_add(view.size as usize);

                let buf = buffers.get(buf_index).ok_or_else(||
                    vortex_err!("view at index {idx} references invalid buffer: {buf_index} out of bounds for VarBinViewArray with {} buffers",
                        buffers.len()))?;

                vortex_ensure!(
                    start_offset < buf.len(),
                    "start offset {start_offset} out of bounds for buffer {buf_index} with size {}",
                    buf.len(),
                );

                vortex_ensure!(
                    end_offset <= buf.len(),
                    "end offset {end_offset} out of bounds for buffer {buf_index} with size {}",
                    buf.len(),
                );

                // Make sure the prefix data matches the buffer data.
                let bytes = &buf[start_offset..end_offset];
                vortex_ensure!(
                    view.prefix == bytes[..4],
                    "VarBinView prefix does not match full string"
                );

                // Validate the full string
                vortex_ensure!(
                    validator(bytes),
                    "view at index {idx}: outlined bytes fails utf-8 validation"
                );
            }
        }

        Ok(())
    }
}

impl VarBinViewArray {
    /// Build a new `VarBinViewArray` from components with validation.
    ///
    /// # Safety
    /// This should only be used when you know for certain that all components are already
    /// validated, for example during array operations that preserve the invariants of the encoding.
    ///
    /// See [`VarBinViewArray::try_new`] for a safe constructor that does validation.
    pub unsafe fn new_unchecked(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        Self {
            dtype,
            buffers,
            views,
            validity,
            stats_set: Default::default(),
        }
    }

    pub fn new(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        Self::try_new(views, buffers, dtype, validity).vortex_expect("VarBinViewArray new")
    }

    pub fn try_new(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&views, &buffers, &dtype, &validity)?;

        Ok(Self {
            dtype,
            buffers,
            views,
            validity,
            stats_set: Default::default(),
        })
    }

    /// Number of raw string data buffers held by this array.
    pub fn nbuffers(&self) -> usize {
        self.buffers.len()
    }

    /// Access to the primitive views buffer.
    ///
    /// Variable-sized binary view buffer contain a "view" child array, with 16-byte entries that
    /// contain either a pointer into one of the array's owned `buffer`s OR an inlined copy of
    /// the string (if the string has 12 bytes or fewer).
    #[inline]
    pub fn views(&self) -> &Buffer<BinaryView> {
        &self.views
    }

    /// Access value bytes at a given index
    ///
    /// Will return a `ByteBuffer` containing the data without performing a copy.
    #[inline]
    pub fn bytes_at(&self, index: usize) -> ByteBuffer {
        let views = self.views();
        let view = &views[index];
        // Expect this to be the common case: strings > 12 bytes.
        if !view.is_inlined() {
            let view_ref = view.as_view();
            self.buffer(view_ref.buffer_index() as usize)
                .slice(view_ref.to_range())
        } else {
            // Return access to the range of bytes around it.
            views
                .clone()
                .into_byte_buffer()
                .slice_ref(view.as_inlined().value())
        }
    }

    /// Access one of the backing data buffers.
    ///
    /// # Panics
    ///
    /// This method panics if the provided index is out of bounds for the set of buffers provided
    /// at construction time.
    #[inline]
    pub fn buffer(&self, idx: usize) -> &ByteBuffer {
        if idx >= self.nbuffers() {
            vortex_panic!(
                "{idx} buffer index out of bounds, there are {} buffers",
                self.nbuffers()
            );
        }
        &self.buffers[idx]
    }

    /// Iterate over the underlying raw data buffers, not including the views buffer.
    #[inline]
    pub fn buffers(&self) -> &Arc<[ByteBuffer]> {
        &self.buffers
    }

    /// Accumulate an iterable set of values into our type here.
    #[allow(clippy::same_name_method)]
    pub fn from_iter<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity(dtype, iter.size_hint().0);

        for item in iter {
            match item {
                None => builder.append_null(),
                Some(v) => builder.append_value(v),
            }
        }

        builder.finish_into_varbinview()
    }

    pub fn from_iter_str<T: AsRef<str>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity(
            DType::Utf8(Nullability::NonNullable),
            iter.size_hint().0,
        );

        for item in iter {
            builder.append_value(item.as_ref());
        }

        builder.finish_into_varbinview()
    }

    pub fn from_iter_nullable_str<T: AsRef<str>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity(
            DType::Utf8(Nullability::Nullable),
            iter.size_hint().0,
        );

        for item in iter {
            match item {
                None => builder.append_null(),
                Some(v) => builder.append_value(v.as_ref()),
            }
        }

        builder.finish_into_varbinview()
    }

    pub fn from_iter_bin<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity(
            DType::Binary(Nullability::NonNullable),
            iter.size_hint().0,
        );

        for item in iter {
            builder.append_value(item.as_ref());
        }

        builder.finish_into_varbinview()
    }

    pub fn from_iter_nullable_bin<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity(
            DType::Binary(Nullability::Nullable),
            iter.size_hint().0,
        );

        for item in iter {
            match item {
                None => builder.append_null(),
                Some(v) => builder.append_value(v.as_ref()),
            }
        }

        builder.finish_into_varbinview()
    }
}

impl ArrayVTable<VarBinViewVTable> for VarBinViewVTable {
    fn len(array: &VarBinViewArray) -> usize {
        array.views.len()
    }

    fn dtype(array: &VarBinViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl ValidityHelper for VarBinViewArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl CanonicalVTable<VarBinViewVTable> for VarBinViewVTable {
    fn canonicalize(array: &VarBinViewArray) -> VortexResult<Canonical> {
        Ok(Canonical::VarBinView(array.clone()))
    }

    fn append_to_builder(
        array: &VarBinViewArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(array.as_ref())
    }
}

impl<'a> FromIterator<Option<&'a [u8]>> for VarBinViewArray {
    fn from_iter<T: IntoIterator<Item = Option<&'a [u8]>>>(iter: T) -> Self {
        Self::from_iter_nullable_bin(iter)
    }
}

impl FromIterator<Option<Vec<u8>>> for VarBinViewArray {
    fn from_iter<T: IntoIterator<Item = Option<Vec<u8>>>>(iter: T) -> Self {
        Self::from_iter_nullable_bin(iter)
    }
}

impl FromIterator<Option<String>> for VarBinViewArray {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        Self::from_iter_nullable_str(iter)
    }
}

impl<'a> FromIterator<Option<&'a str>> for VarBinViewArray {
    fn from_iter<T: IntoIterator<Item = Option<&'a str>>>(iter: T) -> Self {
        Self::from_iter_nullable_str(iter)
    }
}

#[cfg(test)]
mod test {
    use vortex_scalar::Scalar;

    use crate::arrays::varbinview::{BinaryView, VarBinViewArray};
    use crate::{Array, Canonical, IntoArray};

    #[test]
    pub fn varbin_view() {
        let binary_arr =
            VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]);
        assert_eq!(binary_arr.len(), 2);
        assert_eq!(binary_arr.scalar_at(0), Scalar::from("hello world"));
        assert_eq!(
            binary_arr.scalar_at(1),
            Scalar::from("hello world this is a long string")
        );
    }

    #[test]
    pub fn slice_array() {
        let binary_arr =
            VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
                .slice(1, 2);
        assert_eq!(
            binary_arr.scalar_at(0),
            Scalar::from("hello world this is a long string")
        );
    }

    #[test]
    pub fn flatten_array() {
        let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);

        let flattened = binary_arr.to_canonical().unwrap();
        assert!(matches!(flattened, Canonical::VarBinView(_)));

        let var_bin = flattened.into_varbinview().unwrap().into_array();
        assert_eq!(var_bin.scalar_at(0), Scalar::from("string1"));
        assert_eq!(var_bin.scalar_at(1), Scalar::from("string2"));
    }

    #[test]
    pub fn binary_view_size_and_alignment() {
        assert_eq!(size_of::<BinaryView>(), 16);
        assert_eq!(align_of::<BinaryView>(), 16);
    }
}
