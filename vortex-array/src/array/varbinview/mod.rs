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
use vortex_buffer::Buffer;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::arrow::FromArrowArray;
use crate::compute::slice;
use crate::encoding::ids;
use crate::stats::StatsSet;
use crate::validity::{ArrayValidity, LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::PrimitiveArrayTrait;
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayVariant,
    IntoCanonical,
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
            size: value.len() as u32,
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
#[repr(C, align(8))]
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
assert_eq_align!(BinaryView, u64);

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

// reminder: views are 16 bytes with 8-byte alignment
pub(crate) const VIEW_SIZE_BYTES: usize = size_of::<BinaryView>();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarBinViewMetadata {
    // Validity metadata
    validity: ValidityMetadata,

    // Length of each buffer. The buffers are primitive byte arrays containing the raw string/binary
    // data referenced by views.
    buffer_lens: Vec<u32>,
}

impl Display for VarBinViewMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

pub struct Buffers<'a> {
    index: u32,
    n_buffers: u32,
    array: &'a VarBinViewArray,
}

impl Iterator for Buffers<'_> {
    type Item = ArrayData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.n_buffers {
            return None;
        }

        let bytes = self.array.buffer(self.index as usize);
        self.index += 1;
        Some(bytes)
    }
}

impl_encoding!("vortex.varbinview", ids::VAR_BIN_VIEW, VarBinView);

impl VarBinViewArray {
    pub fn try_new(
        views: ArrayData,
        buffers: Vec<ArrayData>,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        if !matches!(views.dtype(), &DType::BYTES) {
            vortex_bail!(MismatchedTypes: "views u8", views.dtype());
        }

        for d in buffers.iter() {
            if !matches!(d.dtype(), &DType::BYTES) {
                vortex_bail!(MismatchedTypes: "buffers u8", d.dtype());
            }
        }

        if !matches!(dtype, DType::Binary(_) | DType::Utf8(_)) {
            vortex_bail!(MismatchedTypes: "utf8 or binary", dtype);
        }

        if dtype.is_nullable() == (validity == Validity::NonNullable) {
            vortex_bail!("incorrect validity {:?}", validity);
        }

        let num_views = views.len() / VIEW_SIZE_BYTES;
        let buffer_lens: Vec<u32> = buffers
            .iter()
            .map(|buffer| -> VortexResult<u32> {
                u32::try_from(buffer.len())
                    .map_err(|e| vortex_err!("buffer must be within 32-bit range: {e}"))
            })
            .try_collect()?;
        let metadata = VarBinViewMetadata {
            validity: validity.to_metadata(num_views)?,
            buffer_lens,
        };

        let mut children = Vec::with_capacity(buffers.len() + 2);
        children.push(views);
        children.extend(buffers);
        if let Some(a) = validity.into_array() {
            children.push(a)
        }

        Self::try_from_parts(
            dtype,
            num_views,
            metadata,
            children.into(),
            StatsSet::default(),
        )
    }

    /// Number of raw string data buffers held by this array.
    pub fn buffer_count(&self) -> usize {
        self.metadata().buffer_lens.len()
    }

    /// Access to the underlying `views` child array as a slice of [BinaryView] structures.
    ///
    /// This is useful for iteration over the values, as well as for applying filters that may
    /// only require hitting the prefixes or inline strings.
    pub fn binary_views(&self) -> VortexResult<Views> {
        Ok(Views {
            inner: self.views().into_primitive()?.into_buffer(),
            idx: 0,
            len: self.len(),
        })
    }

    /// Retrieve a binary view at a specific index.
    ///
    /// If you will be calling this method many times, it's recommended that you instead use the
    /// iterator provided by [`binary_views`][Self::binary_views].
    pub fn view_at(&self, index: usize) -> VortexResult<BinaryView> {
        let start = index * VIEW_SIZE_BYTES;
        let stop = (index + 1) * VIEW_SIZE_BYTES;
        let view_bytes = slice(self.views(), start, stop)?
            .into_primitive()?
            .into_buffer();

        let mut le_bytes: [u8; 16] = [0u8; 16];
        le_bytes.copy_from_slice(view_bytes.as_slice());

        Ok(BinaryView { le_bytes })
    }

    /// Access to the primitive views array.
    ///
    /// Variable-sized binary view arrays contain a "view" child array, with 16-byte entries that
    /// contain either a pointer into one of the array's owned `buffer`s OR an inlined copy of
    /// the string (if the string has 12 bytes or fewer).
    #[inline]
    pub fn views(&self) -> ArrayData {
        self.as_ref()
            .child(0, &DType::BYTES, self.len() * VIEW_SIZE_BYTES)
            .vortex_expect("VarBinViewArray: views child")
    }

    /// Access one of the backing data buffers.
    ///
    /// # Panics
    ///
    /// This method panics if the provided index is out of bounds for the set of buffers provided
    /// at construction time.
    #[inline]
    pub fn buffer(&self, idx: usize) -> ArrayData {
        self.as_ref()
            .child(
                idx + 1,
                &DType::BYTES,
                self.metadata().buffer_lens[idx] as usize,
            )
            .vortex_expect("VarBinViewArray: buffer child")
    }

    /// Retrieve an iterator over the raw data buffers.
    /// These are the BYTE buffers that make up the array's contents.
    ///
    /// Example
    ///
    /// ```
    /// use vortex_array::array::VarBinViewArray;
    /// let array = VarBinViewArray::from_iter_str(["a", "b", "c"]);
    /// array.buffers().for_each(|block| {
    ///     // Do something with the `block`
    /// });
    /// ```
    pub fn buffers(&self) -> Buffers {
        Buffers {
            index: 0,
            n_buffers: u32::try_from(self.buffer_count())
                .unwrap_or_else(|e| vortex_panic!("n_buffers exceeds u32::MAX: {e}")),
            array: self,
        }
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(
                    (self.metadata().buffer_lens.len() + 1) as _,
                    &Validity::DTYPE,
                    self.len(),
                )
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

    // TODO(aduffy): do we really need to do this with copying?
    pub fn bytes_at(&self, index: usize) -> VortexResult<Vec<u8>> {
        let view = self.view_at(index)?;
        // Expect this to be the common case: strings > 12 bytes.
        if !view.is_inlined() {
            let view_ref = view.as_view();
            let data_buf = slice(
                self.buffer(view_ref.buffer_index() as usize),
                view_ref.offset() as usize,
                (view.len() + view_ref.offset()) as usize,
            )?
            .into_primitive()?;
            Ok(data_buf.maybe_null_slice::<u8>().to_vec())
        } else {
            // Return access to the range of bytes around it.
            Ok(view.as_inlined().value().to_vec())
        }
    }
}

pub struct Views {
    inner: Buffer,
    len: usize,
    idx: usize,
}

impl Iterator for Views {
    type Item = BinaryView;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            None
        } else {
            // SAFETY: constructing the BinaryView against u128 value
            let mut le_bytes = [0u8; 16];
            let start = self.idx * VIEW_SIZE_BYTES;
            let stop = start + VIEW_SIZE_BYTES;
            le_bytes.copy_from_slice(&self.inner[start..stop]);

            self.idx += 1;
            Some(BinaryView { le_bytes })
        }
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

impl ArrayTrait for VarBinViewArray {}

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
    // Views should be buffer of u8
    let views = var_bin_view
        .views()
        .into_primitive()
        .vortex_expect("VarBinViewArray: views child must be primitive");
    assert_eq!(views.ptype(), PType::U8);

    let nulls = var_bin_view
        .logical_validity()
        .to_null_buffer()
        .vortex_expect("VarBinViewArray: validity child must be bool");

    let data = (0..var_bin_view.buffer_count())
        .map(|i| var_bin_view.buffer(i).into_primitive())
        .collect::<VortexResult<Vec<_>>>()
        .vortex_expect("VarBinViewArray: bytes children must be primitive");
    if !data.is_empty() {
        assert_eq!(data[0].ptype(), PType::U8);
        assert!(data.iter().map(|d| d.ptype()).all_equal());
    }

    let data = data
        .iter()
        .map(|p| p.buffer().clone().into_arrow())
        .collect::<Vec<_>>();

    // Switch on Arrow DType.
    match var_bin_view.dtype() {
        DType::Binary(_) => Arc::new(unsafe {
            BinaryViewArray::new_unchecked(
                ScalarBuffer::<u128>::from(views.buffer().clone().into_arrow()),
                data,
                nulls,
            )
        }),
        DType::Utf8(_) => Arc::new(unsafe {
            StringViewArray::new_unchecked(
                ScalarBuffer::<u128>::from(views.buffer().clone().into_arrow()),
                data,
                nulls,
            )
        }),
        _ => vortex_panic!("expected utf8 or binary, got {}", var_bin_view.dtype()),
    }
}

impl ValidityVTable<VarBinViewArray> for VarBinViewEncoding {
    fn is_valid(&self, array: &VarBinViewArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &VarBinViewArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<VarBinViewArray> for VarBinViewEncoding {
    fn accept(&self, array: &VarBinViewArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("views", &array.views())?;
        for i in 0..array.metadata().buffer_lens.len() {
            visitor.visit_child(format!("bytes_{i}").as_str(), &array.buffer(i))?;
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

    use crate::array::varbinview::{BinaryView, VarBinViewArray, VIEW_SIZE_BYTES};
    use crate::compute::slice;
    use crate::compute::unary::scalar_at;
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
        assert_eq!(size_of::<BinaryView>(), VIEW_SIZE_BYTES);
        assert_eq!(size_of::<BinaryView>(), 16);
        assert_eq!(align_of::<BinaryView>(), 8);
    }
}
