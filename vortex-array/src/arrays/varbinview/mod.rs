use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;

use arrow_array::builder::{BinaryViewBuilder, GenericByteViewBuilder, StringViewBuilder};
use arrow_array::types::{BinaryViewType, ByteViewType, StringViewType};
use arrow_array::{
    ArrayRef as ArrowArrayRef, BinaryViewArray, GenericByteViewArray, StringViewArray,
};
use arrow_buffer::ScalarBuffer;
use static_assertions::{assert_eq_align, assert_eq_size};
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_panic};
use vortex_mask::Mask;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::arrow::FromArrowArray;
use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::VTableRef;
use crate::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, Canonical, EmptyMetadata, Encoding,
    TryFromArrayRef, try_from_array_ref,
};

mod accessor;
mod compute;
mod serde;
mod variants;

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

#[derive(Clone, Debug)]
pub struct VarBinViewArray {
    dtype: DType,
    buffers: Vec<ByteBuffer>,
    views: Buffer<BinaryView>,
    validity: Validity,
    stats_set: ArrayStats,
}

try_from_array_ref!(VarBinViewArray);

pub struct VarBinViewEncoding;
impl Encoding for VarBinViewEncoding {
    type Array = VarBinViewArray;
    type Metadata = EmptyMetadata;
}

impl VarBinViewArray {
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

    /// Validity of the array
    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Accumulate an iterable set of values into our type here.
    #[allow(clippy::same_name_method)]
    pub fn from_iter<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        match dtype {
            DType::Utf8(nullability) => {
                let string_view_array = generic_byte_view_builder::<StringViewType, _, _>(
                    iter.into_iter(),
                    |builder, v| {
                        match v {
                            None => builder.append_null(),
                            Some(inner) => {
                                // SAFETY: the caller must provide valid utf8 values if Utf8 DType is passed.
                                let utf8 = unsafe { std::str::from_utf8_unchecked(inner.as_ref()) };
                                builder.append_value(utf8);
                            }
                        }
                    },
                );
                VarBinViewArray::try_from_array(ArrayRef::from_arrow(
                    &string_view_array,
                    nullability.into(),
                ))
                .vortex_expect("StringViewArray to VarBinViewArray downcast")
            }
            DType::Binary(nullability) => {
                let binary_view_array = generic_byte_view_builder::<BinaryViewType, _, _>(
                    iter.into_iter(),
                    GenericByteViewBuilder::append_option,
                );
                VarBinViewArray::try_from_array(ArrayRef::from_arrow(
                    &binary_view_array,
                    nullability.into(),
                ))
                .vortex_expect("BinaryViewArray to VarBinViewArray downcast")
            }
            other => vortex_panic!("VarBinViewArray must be Utf8 or Binary, was {other}"),
        }
    }

    pub fn from_iter_str<T: AsRef<str>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = StringViewBuilder::with_capacity(iter.size_hint().0);
        for s in iter {
            builder.append_value(s);
        }
        let array = ArrayRef::from_arrow(&builder.finish(), false);
        VarBinViewArray::try_from_array(array)
            .vortex_expect("VarBinViewArray from StringViewBuilder")
    }

    pub fn from_iter_nullable_str<T: AsRef<str>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = StringViewBuilder::with_capacity(iter.size_hint().0);
        builder.extend(iter);

        let array = ArrayRef::from_arrow(&builder.finish(), true);
        VarBinViewArray::try_from_array(array)
            .vortex_expect("VarBinViewArray from StringViewBuilder")
    }

    pub fn from_iter_bin<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = BinaryViewBuilder::with_capacity(iter.size_hint().0);
        for b in iter {
            builder.append_value(b);
        }
        let array = ArrayRef::from_arrow(&builder.finish(), false);
        VarBinViewArray::try_from_array(array)
            .vortex_expect("VarBinViewArray from StringViewBuilder")
    }

    pub fn from_iter_nullable_bin<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = BinaryViewBuilder::with_capacity(iter.size_hint().0);
        builder.extend(iter);
        let array = ArrayRef::from_arrow(&builder.finish(), true);
        VarBinViewArray::try_from_array(array)
            .vortex_expect("VarBinViewArray from StringViewBuilder")
    }
}

// Generic helper to create an Arrow ByteViewBuilder of the appropriate type.
fn generic_byte_view_builder<B, V, F>(
    values: impl Iterator<Item = Option<V>>,
    mut append_fn: F,
) -> GenericByteViewArray<B>
where
    B: ByteViewType,
    V: AsRef<[u8]>,
    F: FnMut(&mut GenericByteViewBuilder<B>, Option<V>),
{
    let mut builder = GenericByteViewBuilder::<B>::new();

    for value in values {
        append_fn(&mut builder, value);
    }

    builder.finish()
}

impl ArrayImpl for VarBinViewArray {
    type Encoding = VarBinViewEncoding;

    fn _len(&self) -> usize {
        self.views.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&VarBinViewEncoding)
    }
}

impl ArrayStatisticsImpl for VarBinViewArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayCanonicalImpl for VarBinViewArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::VarBinView(self.clone()))
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(self)
    }
}

pub(crate) fn varbinview_as_arrow(var_bin_view: &VarBinViewArray) -> ArrowArrayRef {
    let views = var_bin_view.views().clone();

    let nulls = var_bin_view
        .validity_mask()
        .vortex_expect("VarBinViewArray: failed to get logical validity")
        .to_null_buffer();

    let data = (0..var_bin_view.nbuffers())
        .map(|i| var_bin_view.buffer(i))
        .collect::<Vec<_>>();

    let data = data
        .into_iter()
        .map(|p| p.clone().into_arrow_buffer())
        .collect::<Vec<_>>();

    // Switch on Arrow DType.
    match var_bin_view.dtype() {
        DType::Binary(_) => Arc::new(unsafe {
            BinaryViewArray::new_unchecked(
                ScalarBuffer::<u128>::from(views.into_byte_buffer().into_arrow_buffer()),
                data,
                nulls,
            )
        }),
        DType::Utf8(_) => Arc::new(unsafe {
            StringViewArray::new_unchecked(
                ScalarBuffer::<u128>::from(views.into_byte_buffer().into_arrow_buffer()),
                data,
                nulls,
            )
        }),
        _ => vortex_panic!("expected utf8 or binary, got {}", var_bin_view.dtype()),
    }
}

impl ArrayValidityImpl for VarBinViewArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_logical(self.len())
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

    use crate::Canonical;
    use crate::array::Array;
    use crate::arrays::varbinview::{BinaryView, VarBinViewArray};
    use crate::compute::{scalar_at, slice};

    #[test]
    pub fn varbin_view() {
        let binary_arr =
            VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]);
        assert_eq!(binary_arr.len(), 2);
        assert_eq!(
            scalar_at(&binary_arr, 0).unwrap(),
            Scalar::from("hello world")
        );
        assert_eq!(
            scalar_at(&binary_arr, 1).unwrap(),
            Scalar::from("hello world this is a long string")
        );
    }

    #[test]
    pub fn slice_array() {
        let binary_arr = slice(
            &VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]),
            1,
            2,
        )
        .unwrap();
        assert_eq!(
            scalar_at(&binary_arr, 0).unwrap(),
            Scalar::from("hello world this is a long string")
        );
    }

    #[test]
    pub fn flatten_array() {
        let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);

        let flattened = binary_arr.to_canonical().unwrap();
        assert!(matches!(flattened, Canonical::VarBinView(_)));

        let var_bin = flattened.into_varbinview().unwrap().into_array();
        assert_eq!(scalar_at(&var_bin, 0).unwrap(), Scalar::from("string1"));
        assert_eq!(scalar_at(&var_bin, 1).unwrap(), Scalar::from("string2"));
    }

    #[test]
    pub fn binary_view_size_and_alignment() {
        assert_eq!(size_of::<BinaryView>(), 16);
        assert_eq!(align_of::<BinaryView>(), 16);
    }
}
