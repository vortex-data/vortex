// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::ops::Range;

use static_assertions::{assert_eq_align, assert_eq_size};
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, VortexUnwrap, vortex_bail, vortex_panic};

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

pub use compact::*;

/// An inlined binary view for values up to 12 bytes, which are stored directly in the view.
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

    /// Returns the inlined value as a byte slice.
    #[inline]
    pub fn value(&self) -> &[u8] {
        &self.data[0..(self.size as usize)]
    }
}

/// A reference to binary data stored in an external buffer for values larger than 12 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
pub struct Ref {
    size: u32,
    prefix: [u8; 4],
    buffer_index: u32,
    offset: u32,
}

impl Ref {
    /// Creates a new reference to binary data in an external buffer.
    pub fn new(size: u32, prefix: [u8; 4], buffer_index: u32, offset: u32) -> Self {
        Self {
            size,
            prefix,
            buffer_index,
            offset,
        }
    }

    /// Returns the index of the buffer containing the referenced data.
    #[inline]
    pub fn buffer_index(&self) -> u32 {
        self.buffer_index
    }

    /// Returns the offset within the buffer where the data starts.
    #[inline]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Returns the 4-byte prefix of the data for comparison purposes.
    #[inline]
    pub fn prefix(&self) -> &[u8; 4] {
        &self.prefix
    }

    /// Converts the offset and size to a range for slicing the buffer.
    #[inline]
    pub fn to_range(&self) -> Range<usize> {
        self.offset as usize..(self.offset + self.size) as usize
    }
}

/// A binary view that can either store data inline (≤12 bytes) or reference external buffers.
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
    /// Maximum size in bytes for values that can be stored inline.
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

    /// Returns the length of the binary data in bytes.
    #[inline]
    pub fn len(&self) -> u32 {
        unsafe { self.inlined.size }
    }

    /// Returns true if the binary data is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() > 0
    }

    /// Returns true if the data is stored inline rather than in an external buffer.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn is_inlined(&self) -> bool {
        self.len() <= (Self::MAX_INLINED_SIZE as u32)
    }

    /// Returns a reference to the inlined data. Only safe to call if `is_inlined()` returns true.
    pub fn as_inlined(&self) -> &Inlined {
        unsafe { &self.inlined }
    }

    /// Returns a reference to the buffer reference. Only safe to call if `is_inlined()` returns false.
    pub fn as_view(&self) -> &Ref {
        unsafe { &self._ref }
    }

    /// Returns the binary view as a u128 value for hashing and comparison.
    pub fn as_u128(&self) -> u128 {
        // SAFETY: binary view always safe to read as u128 LE bytes
        unsafe { u128::from_le_bytes(self.le_bytes) }
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

/// A variable-size binary view array supporting both UTF-8 strings and arbitrary binary data.
///
/// This array type uses Apache Arrow's binary view format which can efficiently store both
/// small values (≤12 bytes) inline and larger values in separate data buffers.
#[derive(Clone, Debug)]
pub struct VarBinViewArray {
    dtype: DType,
    buffers: Vec<ByteBuffer>,
    views: Buffer<BinaryView>,
    validity: Validity,
    stats_set: ArrayStats,
}

/// The encoding for variable binary view arrays.
#[derive(Clone, Debug)]
pub struct VarBinViewEncoding;

impl VarBinViewArray {
    /// Creates a new VarBinViewArray from the provided components.
    ///
    /// # Arguments
    ///
    /// * `views` - Buffer of binary views, must be aligned to 128 bits
    /// * `buffers` - Vector of data buffers for values >12 bytes
    /// * `dtype` - Must be Binary or Utf8 type
    /// * `validity` - Validity information for nullable arrays
    pub fn try_new(
        views: Buffer<BinaryView>,
        buffers: Vec<ByteBuffer>,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        if views.alignment() != Alignment::of::<BinaryView>() {
            vortex_bail!("Views must be aligned to a 128 bits");
        }

        if !matches!(dtype, DType::Binary(_) | DType::Utf8(_)) {
            vortex_bail!(MismatchedTypes: "utf8 or binary", dtype);
        }

        if dtype.is_nullable() == (validity == Validity::NonNullable) {
            vortex_bail!("incorrect validity {:?}", validity);
        }

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
    /// Will return a bytebuffer pointing to the underlying data without performing a copy
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
    pub fn buffers(&self) -> &[ByteBuffer] {
        &self.buffers
    }

    /// Creates a VarBinViewArray from an iterator of optional byte values.
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator yielding optional byte arrays
    /// * `dtype` - Must be Binary or Utf8 type
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

    /// Creates a VarBinViewArray from an iterator of string values.
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

    /// Creates a VarBinViewArray from an iterator of optional string values.
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

    /// Creates a VarBinViewArray from an iterator of binary values.
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

    /// Creates a VarBinViewArray from an iterator of optional binary values.
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
        assert_eq!(
            binary_arr.scalar_at(0).unwrap(),
            Scalar::from("hello world")
        );
        assert_eq!(
            binary_arr.scalar_at(1).unwrap(),
            Scalar::from("hello world this is a long string")
        );
    }

    #[test]
    pub fn slice_array() {
        let binary_arr =
            VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
                .slice(1, 2)
                .unwrap();
        assert_eq!(
            binary_arr.scalar_at(0).unwrap(),
            Scalar::from("hello world this is a long string")
        );
    }

    #[test]
    pub fn flatten_array() {
        let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);

        let flattened = binary_arr.to_canonical().unwrap();
        assert!(matches!(flattened, Canonical::VarBinView(_)));

        let var_bin = flattened.into_varbinview().unwrap().into_array();
        assert_eq!(var_bin.scalar_at(0).unwrap(), Scalar::from("string1"));
        assert_eq!(var_bin.scalar_at(1).unwrap(), Scalar::from("string2"));
    }

    #[test]
    pub fn binary_view_size_and_alignment() {
        assert_eq!(size_of::<BinaryView>(), 16);
        assert_eq!(align_of::<BinaryView>(), 16);
    }
}
