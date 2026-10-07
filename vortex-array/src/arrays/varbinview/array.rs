// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::mem::size_of;
use std::sync::Arc;

use smallvec::smallvec;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionCtx;
#[cfg(debug_assertions)]
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::array_slots;
use crate::arrays::VarBinView;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::ResolvedViews;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
#[cfg(debug_assertions)]
use crate::legacy_session;
use crate::validity::Validity;

#[array_slots(VarBinView)]
pub struct VarBinViewSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity: Option<ArrayRef>,
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
/// use vortex_array::dtype::{DType, Nullability};
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
pub struct VarBinViewData {
    pub(super) buffers: Arc<[BufferHandle]>,
    pub(super) views: BufferHandle,
}

impl Display for VarBinViewData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct VarBinViewDataParts {
    pub dtype: DType,
    pub buffers: Arc<[BufferHandle]>,
    pub views: BufferHandle,
    pub validity: Validity,
}

// Walk invalid (null) runs of "mask" as [start, end) ranges
fn for_each_invalid_range(mask: &Mask, len: usize, mut f: impl FnMut(usize, usize)) {
    match mask.bit_buffer() {
        AllOr::All => {}
        AllOr::None => f(0, len),
        AllOr::Some(buffer) => {
            let mut prev = 0;
            for (start, end) in buffer.set_slices() {
                if start > prev {
                    f(prev, start);
                }
                prev = end;
            }
            if prev < len {
                f(prev, len);
            }
        }
    }
}

/// Calls `f` for each valid index of `mask`, and stops at the first error.
fn try_for_each_valid(
    mask: &Mask,
    mut f: impl FnMut(usize) -> VortexResult<()>,
) -> VortexResult<()> {
    match mask.bit_buffer() {
        AllOr::All => (0..mask.len()).try_for_each(f),
        AllOr::None => Ok(()),
        AllOr::Some(buffer) => buffer.set_indices().try_for_each(&mut f),
    }
}

/// Returns `true` if all the inlined bytes of `view` are ASCII.
fn is_inlined_ascii(view: &BinaryView) -> bool {
    const HIGH_BITS: u128 = u128::from_le_bytes([0x80; 16]);

    // The first 4 bytes hold the length, and the inlined bytes follow.
    let data_bits = ((1u128 << (8 * view.len())) - 1) << 32;
    view.as_u128() & data_bits & HIGH_BITS == 0
}

/// Returns `true` if `byte` starts a UTF-8 char, that is, it is not a continuation byte.
fn is_char_boundary(byte: u8) -> bool {
    // Continuation bytes have the form `0b10xx_xxxx`.
    byte & 0b1100_0000 != 0b1000_0000
}

impl VarBinViewData {
    fn dtype_parts(dtype: &DType) -> VortexResult<(bool, Nullability)> {
        match dtype {
            DType::Utf8(nullability) => Ok((true, *nullability)),
            DType::Binary(nullability) => Ok((false, *nullability)),
            _ => vortex_bail!(InvalidArgument: "invalid DType {dtype} for `VarBinViewArray`"),
        }
    }

    /// Build the slots vector for this array.
    pub(super) fn make_slots(validity: &Validity, len: usize) -> ArraySlots {
        smallvec![validity_to_child(validity, len)]
    }

    /// Creates a new `VarBinViewArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `VarBinViewArray::new_unchecked`.
    pub fn new(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::try_new(views, buffers, dtype, validity, ctx)
            .vortex_expect("VarBinViewArray construction failed")
    }

    /// Creates a new `VarBinViewArray` with device or host memory.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `VarBinViewArray::new_unchecked`.
    pub fn new_handle(
        views: BufferHandle,
        buffers: Arc<[BufferHandle]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        Self::try_new_handle(views, buffers, dtype, validity)
            .vortex_expect("VarbinViewArray construction failed")
    }

    /// Constructs a new `VarBinViewArray`.
    ///
    /// See `VarBinViewArray::new_unchecked` for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// `VarBinViewArray::new_unchecked`.
    pub fn try_new(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let views = Self::validate_and_fix(views, &buffers, &dtype, &validity, ctx)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(views, buffers, dtype, validity) })
    }

    /// Constructs a new `VarBinViewArray`.
    ///
    /// See `VarBinViewArray::new_unchecked` for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// `VarBinViewArray::new_unchecked`.
    pub fn try_new_handle(
        views: BufferHandle,
        buffers: Arc<[BufferHandle]>,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        let views_nbytes = views.len();
        vortex_ensure!(
            views_nbytes.is_multiple_of(size_of::<BinaryView>()),
            "Expected views buffer length ({views_nbytes}) to be a multiple of {}",
            size_of::<BinaryView>()
        );

        // TODO(aduffy): device validation.
        if let Some(host) = views.as_host_opt() {
            vortex_ensure!(
                host.is_aligned(Alignment::of::<BinaryView>()),
                "Views on host must be 16 byte aligned"
            );
        }

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_handle_unchecked(views, buffers, dtype, validity) })
    }

    /// Creates a new `VarBinViewArray` without validation from these components:
    ///
    /// * `views` is a buffer of 16-byte view entries (one per logical element).
    /// * `buffers` contains the backing storage for strings longer than 12 bytes.
    /// * `dtype` specifies whether this contains UTF-8 strings or binary data.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// ## View Requirements
    ///
    /// - Views must be properly formatted 16-byte [`BinaryView`] entries.
    /// - Inlined views (length ≤ 12) must have valid data in the first `length` bytes.
    /// - Reference views (length > 12) must:
    ///   - Have a valid buffer index < `buffers.len()`.
    ///   - Have valid offsets that don't exceed the referenced buffer's bounds.
    ///   - Have a 4-byte prefix that matches the actual data at the referenced location.
    ///
    /// ## Type Requirements
    ///
    /// - `dtype` must be either [`DType::Utf8`] or [`DType::Binary`].
    /// - For [`DType::Utf8`], all string data (both inlined and referenced) must be valid UTF-8.
    ///
    /// ## Validity Requirements
    ///
    /// - The validity must have the same nullability as the dtype.
    /// - If validity is an array, its length must match `views.len()`.
    pub unsafe fn new_unchecked(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        #[cfg(debug_assertions)]
        #[expect(clippy::disallowed_methods)]
        Self::validate(
            &views,
            &buffers,
            &dtype,
            &validity,
            &mut legacy_session().create_execution_ctx(),
        )
        .vortex_expect("[Debug Assertion]: Invalid `VarBinViewArray` parameters");

        let handles: Vec<BufferHandle> = buffers
            .iter()
            .cloned()
            .map(BufferHandle::new_host)
            .collect();

        let handles = Arc::from(handles);
        let view_handle = BufferHandle::new_host(views.into_byte_buffer());
        unsafe { Self::new_handle_unchecked(view_handle, handles, dtype, validity) }
    }

    /// Construct a new array from `BufferHandle`s without validation.
    ///
    /// # Safety
    ///
    /// See documentation in `new_unchecked`.
    pub unsafe fn new_handle_unchecked(
        views: BufferHandle,
        buffers: Arc<[BufferHandle]>,
        dtype: DType,
        _validity: Validity,
    ) -> Self {
        let _ =
            Self::dtype_parts(&dtype).vortex_expect("VarBinViewArray dtype must be utf8 or binary");
        Self { buffers, views }
    }

    /// Validates the components that would be used to create a `VarBinViewArray`.
    ///
    /// This function checks all the invariants required by `VarBinViewArray::new_unchecked`,
    /// validating only the views at non-null slots.
    pub fn validate(
        views: &Buffer<BinaryView>,
        buffers: &Arc<[ByteBuffer]>,
        dtype: &DType,
        validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let check_utf8 = Self::check_nullability(dtype, validity)?;
        let mask = validity.execute_mask(views.len(), ctx)?;
        Self::validate_views(views, &mask, buffers, check_utf8)
    }

    /// Validates components like validate() and replaces views at null slots to empty views
    pub(crate) fn validate_and_fix(
        views: Buffer<BinaryView>,
        buffers: &Arc<[ByteBuffer]>,
        dtype: &DType,
        validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Buffer<BinaryView>> {
        let check_utf8 = Self::check_nullability(dtype, validity)?;
        let mask = validity.execute_mask(views.len(), ctx)?;

        // With nulls and views we own, validate and replace in a single pass: a second pass over
        // many short null runs costs more than a branch per view.
        let views = match (mask.bit_buffer(), views.try_into_mut()) {
            (AllOr::Some(bits), Ok(mut views)) => {
                let buffers_utf8 = Self::buffers_utf8(&mask, buffers, check_utf8);
                let buffers_utf8 = buffers_utf8.as_deref();
                let empty = BinaryView::empty_view();
                let slice = views.as_mut_slice();
                for (idx, (view, valid)) in slice.iter_mut().zip(bits.iter()).enumerate() {
                    if valid {
                        Self::validate_view(idx, view, buffers, buffers_utf8)?;
                    } else {
                        *view = empty;
                    }
                }
                return Ok(views.freeze());
            }
            (_, Ok(views)) => views.freeze(),
            (_, Err(views)) => views,
        };

        Self::validate_views(&views, &mask, buffers, check_utf8)?;
        Ok(Self::replace_null_views(views, &mask))
    }

    fn check_nullability(dtype: &DType, validity: &Validity) -> VortexResult<bool> {
        let (is_utf8, nullability) = Self::dtype_parts(dtype)?;
        vortex_ensure_eq!(
            validity.nullability(),
            nullability,
            InvalidArgument: "validity nullability is incompatible with dtype nullability",
        );
        Ok(is_utf8)
    }

    /// Replaces the views at null slots with empty views. Copies the views only if a null slot
    /// holds a non-empty view and the buffer is shared.
    fn replace_null_views(views: Buffer<BinaryView>, mask: &Mask) -> Buffer<BinaryView> {
        if mask.all_true() {
            return views;
        }

        let empty = BinaryView::empty_view();
        let len = views.len();

        let mut views = match views.try_into_mut() {
            Ok(views) => views,
            Err(views) => {
                let mut needs_replace = false;
                for_each_invalid_range(mask, len, |start, end| {
                    needs_replace |= views[start..end].iter().any(|view| *view != empty);
                });
                if !needs_replace {
                    return views;
                }
                views.into_mut()
            }
        };

        let slice = views.as_mut_slice();
        for_each_invalid_range(mask, len, |start, end| slice[start..end].fill(empty));
        views.freeze()
    }

    /// Validates the views at the valid slots of `mask`.
    ///
    /// A per-view UTF-8 check costs a function call per view, which dominates for short strings.
    /// To avoid it, each data buffer is first checked once as a whole. If a buffer is valid UTF-8,
    /// a view into it is valid UTF-8 if and only if it starts and ends on a char boundary.
    ///
    /// A buffer can hold bytes that no valid view references, for example after a slice, after a
    /// filter, or at null views. So the whole-buffer check is only done while the total size of
    /// the checked buffers stays within a fixed cost per valid view. The views into a buffer that
    /// is not checked, or that is not valid UTF-8, are checked one by one.
    fn validate_views(
        views: &[BinaryView],
        mask: &Mask,
        buffers: &[ByteBuffer],
        check_utf8: bool,
    ) -> VortexResult<()> {
        let buffers_utf8 = Self::buffers_utf8(mask, buffers, check_utf8);
        let buffers_utf8 = buffers_utf8.as_deref();

        try_for_each_valid(mask, |idx| {
            Self::validate_view(idx, &views[idx], buffers, buffers_utf8)
        })
    }

    /// The approximate cost of a per-view UTF-8 check, in bytes of a whole-buffer check.
    const PER_VIEW_UTF8_COST: usize = 64;

    /// Returns `None` if the UTF-8 check is off. Otherwise, returns for each buffer `true` if the
    /// whole buffer was checked and is valid UTF-8.
    ///
    /// The budget only counts a fixed cost per valid view, not the bytes the views reference: for
    /// long strings, the call per view is cheap next to the string, and checking a buffer larger
    /// than the cache and then reading it again for the char boundaries is slower.
    fn buffers_utf8(mask: &Mask, buffers: &[ByteBuffer], check_utf8: bool) -> Option<Vec<bool>> {
        if !check_utf8 {
            return None;
        }

        let mut budget = mask.true_count().saturating_mul(Self::PER_VIEW_UTF8_COST);

        Some(
            buffers
                .iter()
                .map(|buf| {
                    if buf.len() > budget {
                        return false;
                    }

                    budget -= buf.len();
                    simdutf8::basic::from_utf8(buf).is_ok()
                })
                .collect(),
        )
    }

    /// Checks the bounds, the prefix, and the UTF-8 of a view.
    ///
    /// `buffers_utf8` is `None` if the UTF-8 check is off. Otherwise, it tells which buffers are
    /// valid UTF-8 as a whole.
    fn validate_view(
        idx: usize,
        view: &BinaryView,
        buffers: &[ByteBuffer],
        buffers_utf8: Option<&[bool]>,
    ) -> VortexResult<()> {
        if view.is_inlined() {
            if buffers_utf8.is_some() && !is_inlined_ascii(view) {
                let bytes = view.as_inlined().value();
                vortex_ensure!(
                    simdutf8::basic::from_utf8(bytes).is_ok(),
                    InvalidArgument: "view at index {idx}: inlined bytes failed utf-8 validation"
                );
            }
            return Ok(());
        }

        let view = view.as_view();
        let buf_index = view.buffer_index as usize;
        let start_offset = view.offset as usize;
        let end_offset = start_offset.saturating_add(view.size as usize);

        let buf = buffers.get(buf_index).ok_or_else(||
            vortex_err!(InvalidArgument: "view at index {idx} references invalid buffer: {buf_index} out of bounds for VarBinViewData with {} buffers",
                buffers.len()))?;

        vortex_ensure!(
            start_offset < buf.len(),
            InvalidArgument: "start offset {start_offset} out of bounds for buffer {buf_index} with size {}",
            buf.len(),
        );

        vortex_ensure!(
            end_offset <= buf.len(),
            InvalidArgument: "end offset {end_offset} out of bounds for buffer {buf_index} with size {}",
            buf.len(),
        );

        // Make sure the prefix data matches the buffer data.
        vortex_ensure!(
            view.prefix == buf[start_offset..start_offset + 4],
            InvalidArgument: "VarBinView prefix does not match full string"
        );

        let Some(buffers_utf8) = buffers_utf8 else {
            return Ok(());
        };

        let valid_utf8 = if buffers_utf8[buf_index] {
            is_char_boundary(buf[start_offset])
                && (end_offset == buf.len() || is_char_boundary(buf[end_offset]))
        } else {
            simdutf8::basic::from_utf8(&buf[start_offset..end_offset]).is_ok()
        };

        vortex_ensure!(
            valid_utf8,
            InvalidArgument: "view at index {idx}: outlined bytes fails utf-8 validation"
        );

        Ok(())
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.views.len() / size_of::<BinaryView>()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Access to the primitive views buffer.
    ///
    /// Variable-sized binary view buffer contain a "view" child array, with 16-byte entries that
    /// contain either a pointer into one of the array's owned `buffer`s OR an inlined copy of
    /// the string (if the string has 12 bytes or fewer).
    #[inline]
    pub fn views(&self) -> &[BinaryView] {
        let host_views = self.views.as_host();
        let len = host_views.len() / size_of::<BinaryView>();

        // SAFETY: data alignment is checked for host buffers on construction
        unsafe { std::slice::from_raw_parts(host_views.as_ptr().cast(), len) }
    }

    /// Return the buffer handle backing the views.
    pub fn views_handle(&self) -> &BufferHandle {
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
            self.buffer(view_ref.buffer_index as usize)
                .slice(view_ref.as_range())
        } else {
            // Return access to the range of bytes around it.
            self.views_handle()
                .as_host()
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
        if idx >= self.data_buffers().len() {
            vortex_panic!(
                "{idx} buffer index out of bounds, there are {} buffers",
                self.data_buffers().len()
            );
        }
        self.buffers[idx].as_host()
    }

    /// The underlying raw data buffers, not including the views buffer.
    #[inline]
    pub fn data_buffers(&self) -> &Arc<[BufferHandle]> {
        &self.buffers
    }

    /// Accumulate an iterable set of values into our type here.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally named from_iter like Iterator::from_iter"
    )]
    pub fn from_iter<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype,
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );

        for item in iter {
            match item {
                None => builder.append_null(),
                Some(v) => builder.append_value(v),
            }
        }

        builder.finish_into_varbinview().into_data()
    }

    pub fn from_iter_str<T: AsRef<str>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::NonNullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );

        for item in iter {
            builder.append_value(item.as_ref());
        }

        builder.finish_into_varbinview().into_data()
    }

    pub fn from_iter_nullable_str<T: AsRef<str>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::Nullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );

        for item in iter {
            match item {
                None => builder.append_null(),
                Some(v) => builder.append_value(v.as_ref()),
            }
        }

        builder.finish_into_varbinview().into_data()
    }

    pub fn from_iter_bin<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Binary(Nullability::NonNullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );

        for item in iter {
            builder.append_value(item.as_ref());
        }

        builder.finish_into_varbinview().into_data()
    }

    pub fn from_iter_nullable_bin<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Binary(Nullability::Nullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );

        for item in iter {
            match item {
                None => builder.append_null(),
                Some(v) => builder.append_value(v.as_ref()),
            }
        }

        builder.finish_into_varbinview().into_data()
    }
}

pub trait VarBinViewArrayExt: TypedArrayRef<VarBinView> {
    fn dtype_parts(&self) -> (bool, Nullability) {
        match self.as_ref().dtype() {
            DType::Utf8(nullability) => (true, *nullability),
            DType::Binary(nullability) => (false, *nullability),
            _ => unreachable!("VarBinViewArrayExt requires a utf8 or binary dtype"),
        }
    }

    fn varbinview_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[VarBinViewSlots::VALIDITY].as_ref(),
            self.dtype_parts().1,
        )
    }
}
impl<T: TypedArrayRef<VarBinView>> VarBinViewArrayExt for T {}

impl Array<VarBinView> {
    /// Resolve the data buffers of this array for per-row access.
    pub fn resolved_views(&self) -> ResolvedViews<'_> {
        ResolvedViews::new(self)
    }

    #[inline]
    fn from_prevalidated_data(dtype: DType, data: VarBinViewData, slots: ArraySlots) -> Self {
        let len = data.len();
        unsafe { Array::from_parts_unchecked(ArrayParts::new(VarBinView, dtype, len, data, slots)) }
    }

    /// Construct a `VarBinViewArray` from an iterator of optional byte slices.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally named from_iter like Iterator::from_iter"
    )]
    pub fn from_iter<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype,
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );
        for value in iter {
            match value {
                Some(value) => builder.append_value(value),
                None => builder.append_null(),
            }
        }
        builder.finish_into_varbinview()
    }

    pub fn from_iter_str<T: AsRef<str>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::NonNullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );
        for value in iter {
            builder.append_value(value.as_ref());
        }
        builder.finish_into_varbinview()
    }

    pub fn from_iter_nullable_str<T: AsRef<str>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Utf8(Nullability::Nullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );
        for value in iter {
            match value {
                Some(value) => builder.append_value(value.as_ref()),
                None => builder.append_null(),
            }
        }
        builder.finish_into_varbinview()
    }

    pub fn from_iter_bin<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Binary(Nullability::NonNullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );
        for value in iter {
            builder.append_value(value.as_ref());
        }
        builder.finish_into_varbinview()
    }

    pub fn from_iter_nullable_bin<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinViewBuilder::with_capacity_in(
            DType::Binary(Nullability::Nullable),
            iter.size_hint().0,
            BufferAllocatorRef::statically_allocated(),
        );
        for value in iter {
            match value {
                Some(value) => builder.append_value(value.as_ref()),
                None => builder.append_null(),
            }
        }
        builder.finish_into_varbinview()
    }

    /// Creates a new `VarBinViewArray`.
    pub fn try_new(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let data = VarBinViewData::try_new(views, buffers, dtype.clone(), validity.clone(), ctx)?;
        let slots = VarBinViewData::make_slots(&validity, data.len());
        Ok(Self::from_prevalidated_data(dtype, data, slots))
    }

    /// Creates a new `VarBinViewArray` without validation.
    ///
    /// # Safety
    ///
    /// See [`VarBinViewData::new_unchecked`].
    pub unsafe fn new_unchecked(
        views: Buffer<BinaryView>,
        buffers: Arc<[ByteBuffer]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        let data = unsafe {
            VarBinViewData::new_unchecked(views, buffers, dtype.clone(), validity.clone())
        };
        let slots = VarBinViewData::make_slots(&validity, data.len());
        Self::from_prevalidated_data(dtype, data, slots)
    }

    /// Creates a new `VarBinViewArray` with device or host memory.
    pub fn new_handle(
        views: BufferHandle,
        buffers: Arc<[BufferHandle]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        let data = VarBinViewData::new_handle(views, buffers, dtype.clone(), validity.clone());
        let slots = VarBinViewData::make_slots(&validity, data.len());
        Self::from_prevalidated_data(dtype, data, slots)
    }

    /// Construct a new array from `BufferHandle`s without validation.
    ///
    /// # Safety
    ///
    /// See [`VarBinViewData::new_handle_unchecked`].
    pub unsafe fn new_handle_unchecked(
        views: BufferHandle,
        buffers: Arc<[BufferHandle]>,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        let data = unsafe {
            VarBinViewData::new_handle_unchecked(views, buffers, dtype.clone(), validity.clone())
        };
        let slots = VarBinViewData::make_slots(&validity, data.len());
        Self::from_prevalidated_data(dtype, data, slots)
    }

    pub fn into_data_parts(self) -> VarBinViewDataParts {
        let dtype = self.dtype().clone();
        let validity = self.varbinview_validity();
        let data = self.into_data();
        VarBinViewDataParts {
            dtype,
            buffers: data.buffers,
            views: data.views,
            validity,
        }
    }
}

impl<'a> FromIterator<Option<&'a [u8]>> for VarBinViewData {
    fn from_iter<T: IntoIterator<Item = Option<&'a [u8]>>>(iter: T) -> Self {
        Self::from_iter_nullable_bin(iter)
    }
}

impl FromIterator<Option<Vec<u8>>> for VarBinViewData {
    fn from_iter<T: IntoIterator<Item = Option<Vec<u8>>>>(iter: T) -> Self {
        Self::from_iter_nullable_bin(iter)
    }
}

impl FromIterator<Option<String>> for VarBinViewData {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        Self::from_iter_nullable_str(iter)
    }
}

impl<'a> FromIterator<Option<&'a str>> for VarBinViewData {
    fn from_iter<T: IntoIterator<Item = Option<&'a str>>>(iter: T) -> Self {
        Self::from_iter_nullable_str(iter)
    }
}

// --- FromIterator forwarding for Array<VarBinView> ---

impl<'a> FromIterator<Option<&'a [u8]>> for Array<VarBinView> {
    fn from_iter<T: IntoIterator<Item = Option<&'a [u8]>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Binary(Nullability::Nullable))
    }
}

impl FromIterator<Option<Vec<u8>>> for Array<VarBinView> {
    fn from_iter<T: IntoIterator<Item = Option<Vec<u8>>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Binary(Nullability::Nullable))
    }
}

impl FromIterator<Option<String>> for Array<VarBinView> {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        Self::from_iter_nullable_str(iter)
    }
}

impl<'a> FromIterator<Option<&'a str>> for Array<VarBinView> {
    fn from_iter<T: IntoIterator<Item = Option<&'a str>>>(iter: T) -> Self {
        Self::from_iter_nullable_str(iter)
    }
}
