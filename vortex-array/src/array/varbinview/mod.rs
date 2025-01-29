use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use ::serde::{Deserialize, Serialize};
use arrow_array::builder::{BinaryViewBuilder, GenericByteViewBuilder, StringViewBuilder};
use arrow_array::types::{BinaryViewType, ByteViewType, StringViewType};
use arrow_array::{ArrayRef, BinaryViewArray, GenericByteViewArray, StringViewArray};
use arrow_buffer::ScalarBuffer;
use itertools::Itertools;
use static_assertions::{assert_eq_align, assert_eq_size};
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_panic, VortexExpect, VortexResult, VortexUnwrap};
use vortex_mask::Mask;

use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validity::{ArrayValidity, Validity, ValidityMetadata};
use crate::visitor::ArrayVisitor;
use crate::vtable::{ValidateVTable, ValidityVTable, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, Canonical, DeserializeMetadata, IntoCanonical,
    RkyvMetadata,
};

mod accessor;
mod compute;
mod stats;
mod variants;

#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
pub struct Inlined {
    size: u32,
    data: [u8; BinaryView::MAX_INLINED_SIZE],
}

impl Inlined {
    pub fn new(value: &[u8]) -> Self {
        assert!(
            value.len() <= BinaryView::MAX_INLINED_SIZE,
            "Inlined strings must be shorter than 13 characters, {} given",
            value.len()
        );
        let mut inlined = Self {
            size: value.len().try_into().vortex_unwrap(),
            data: [0u8; BinaryView::MAX_INLINED_SIZE],
        };
        inlined.data[..value.len()].copy_from_slice(value);
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

    pub fn new_inlined(value: &[u8]) -> Self {
        assert!(
            value.len() <= Self::MAX_INLINED_SIZE,
            "expected inlined value to be <= 12 bytes, was {}",
            value.len()
        );

        Self {
            inlined: Inlined::new(value),
        }
    }

    /// Create a new view over bytes stored in a block.
    pub fn new_view(len: u32, prefix: [u8; 4], block: u32, offset: u32) -> Self {
        Self {
            _ref: Ref::new(len, prefix, block, offset),
        }
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

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct VarBinViewMetadata {
    // Validity metadata
    pub(crate) validity: ValidityMetadata,
}

impl Display for VarBinViewMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl_encoding!(
    "vortex.varbinview",
    ids::VAR_BIN_VIEW,
    VarBinView,
    RkyvMetadata<VarBinViewMetadata>
);

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

        let metadata = VarBinViewMetadata {
            validity: validity.to_metadata(views.len())?,
        };

        let array_len = views.len();

        let mut array_buffers = Vec::with_capacity(buffers.len() + 1);
        array_buffers.push(views.into_byte_buffer());
        array_buffers.extend(buffers);

        Self::try_from_parts(
            dtype,
            array_len,
            RkyvMetadata(metadata),
            Some(array_buffers.into()),
            validity.into_array().map(|v| [v].into()),
            StatsSet::default(),
        )
    }

    /// Number of raw string data buffers held by this array.
    pub fn nbuffers(&self) -> usize {
        self.0.nbuffers() - 1
    }

    /// Access to the primitive views buffer.
    ///
    /// Variable-sized binary view buffer contain a "view" child array, with 16-byte entries that
    /// contain either a pointer into one of the array's owned `buffer`s OR an inlined copy of
    /// the string (if the string has 12 bytes or fewer).
    #[inline]
    pub fn views(&self) -> Buffer<BinaryView> {
        Buffer::from_byte_buffer(
            self.0
                .byte_buffer(0)
                .vortex_expect("Expected a views buffer")
                .clone(),
        )
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
    pub fn buffer(&self, idx: usize) -> ByteBuffer {
        if idx >= self.nbuffers() {
            vortex_panic!(
                "{idx} buffer index out of bounds, there are {} buffers",
                self.nbuffers()
            );
        }

        self.0
            .byte_buffer(idx + 1)
            .vortex_expect("Out of bounds view buffer")
            .clone()
    }

    /// Iterate over the underlying raw data buffers, not including the views buffer.
    #[inline]
    pub fn buffers(&self) -> impl Iterator<Item = ByteBuffer> + '_ {
        self.0.byte_buffers().skip(1)
    }

    /// Validity of the array
    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.0
                .child(0, &Validity::DTYPE, self.len())
                .vortex_expect("VarBinViewArray: validity child")
        })
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
                VarBinViewArray::try_from(ArrayData::from_arrow(
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
                VarBinViewArray::try_from(ArrayData::from_arrow(
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
        let array = ArrayData::from_arrow(&builder.finish(), false);
        VarBinViewArray::try_from(array).vortex_expect("VarBinViewArray from StringViewBuilder")
    }

    pub fn from_iter_nullable_str<T: AsRef<str>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = StringViewBuilder::with_capacity(iter.size_hint().0);
        builder.extend(iter);

        let array = ArrayData::from_arrow(&builder.finish(), true);
        VarBinViewArray::try_from(array).vortex_expect("VarBinViewArray from StringViewBuilder")
    }

    pub fn from_iter_bin<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = BinaryViewBuilder::with_capacity(iter.size_hint().0);
        for b in iter {
            builder.append_value(b);
        }
        let array = ArrayData::from_arrow(&builder.finish(), false);
        VarBinViewArray::try_from(array).vortex_expect("VarBinViewArray from StringViewBuilder")
    }

    pub fn from_iter_nullable_bin<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = BinaryViewBuilder::with_capacity(iter.size_hint().0);
        builder.extend(iter);
        let array = ArrayData::from_arrow(&builder.finish(), true);
        VarBinViewArray::try_from(array).vortex_expect("VarBinViewArray from StringViewBuilder")
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

impl ValidateVTable<VarBinViewArray> for VarBinViewEncoding {}

impl IntoCanonical for VarBinViewArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let nullable = self.dtype().is_nullable();
        let arrow_self = varbinview_as_arrow(&self);
        let vortex_array = ArrayData::from_arrow(arrow_self, nullable);

        Ok(Canonical::VarBinView(VarBinViewArray::try_from(
            vortex_array,
        )?))
    }
}

pub(crate) fn varbinview_as_arrow(var_bin_view: &VarBinViewArray) -> ArrayRef {
    let views = var_bin_view.views();

    let nulls = var_bin_view
        .logical_validity()
        .vortex_expect("VarBinViewArray: failed to get logical validity")
        .to_null_buffer();

    let data = (0..var_bin_view.nbuffers())
        .map(|i| var_bin_view.buffer(i))
        .collect::<Vec<_>>();

    let data = data
        .iter()
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

impl ValidityVTable<VarBinViewArray> for VarBinViewEncoding {
    fn is_valid(&self, array: &VarBinViewArray, index: usize) -> VortexResult<bool> {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &VarBinViewArray) -> VortexResult<Mask> {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<VarBinViewArray> for VarBinViewEncoding {
    fn accept(&self, array: &VarBinViewArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        for buffer in array.as_ref().byte_buffers() {
            visitor.visit_buffer(&buffer)?;
        }

        visitor.visit_validity(&array.validity())
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

    use crate::array::varbinview::{BinaryView, VarBinViewArray};
    use crate::compute::{scalar_at, slice};
    use crate::{ArrayLen, Canonical, IntoArrayData, IntoCanonical};

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
            VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
                .into_array(),
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

        let flattened = binary_arr.into_canonical().unwrap();
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
