// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::DynArray;
use crate::ToCanonical;
use crate::arrays::varbin::builder::VarBinBuilder;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::validity_to_child;

pub(super) const OFFSETS_SLOT: usize = 0;
pub(super) const VALIDITY_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["offsets", "validity"];

#[derive(Clone, Debug)]
pub struct VarBinArray {
    pub(super) dtype: DType,
    pub(super) bytes: BufferHandle,
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

impl VarBinArray {
    /// Creates a new [`VarBinArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`VarBinArray::new_unchecked`].
    pub fn new(offsets: ArrayRef, bytes: ByteBuffer, dtype: DType, validity: Validity) -> Self {
        Self::try_new(offsets, bytes, dtype, validity).vortex_expect("VarBinArray new")
    }

    /// Creates a new [`VarBinArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`VarBinArray::new_unchecked`].
    pub fn new_from_handle(
        offset: ArrayRef,
        bytes: BufferHandle,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        Self::try_new_from_handle(offset, bytes, dtype, validity).vortex_expect("VarBinArray new")
    }

    /// Constructs a new `VarBinArray`.
    ///
    /// See [`VarBinArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`VarBinArray::new_unchecked`].
    pub fn try_new(
        offsets: ArrayRef,
        bytes: ByteBuffer,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        let bytes = BufferHandle::new_host(bytes);
        Self::validate(&offsets, &bytes, &dtype, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked_from_handle(offsets, bytes, dtype, validity) })
    }

    /// Constructs a new `VarBinArray` from a `BufferHandle` of memory that may exist
    /// on the CPU or GPU.
    ///
    /// See [`VarBinArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`VarBinArray::new_unchecked`].
    pub fn try_new_from_handle(
        offsets: ArrayRef,
        bytes: BufferHandle,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&offsets, &bytes, &dtype, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked_from_handle(offsets, bytes, dtype, validity) })
    }

    /// Creates a new [`VarBinArray`] without validation from these components:
    ///
    /// * `offsets` is an array of byte offsets into the `bytes` buffer.
    /// * `bytes` is a buffer containing all the variable-length data concatenated.
    /// * `dtype` specifies whether this contains UTF-8 strings or binary data.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// ## Offsets Requirements
    ///
    /// - `offsets` must be a non-nullable integer array.
    /// - `offsets` must contain at least 1 element (for empty array, it contains \[0\]).
    /// - All values in `offsets` must be monotonically non-decreasing.
    /// - The first value in `offsets` must be 0.
    /// - No offset value may exceed `bytes.len()`.
    ///
    /// ## Type Requirements
    ///
    /// - `dtype` must be exactly [`DType::Binary`] or [`DType::Utf8`].
    /// - If `dtype` is [`DType::Utf8`], every byte slice `bytes[offsets[i]..offsets[i+1]]` must be valid UTF-8.
    /// - `dtype.is_nullable()` must match the nullability of `validity`.
    ///
    /// ## Validity Requirements
    ///
    /// - If `validity` is [`Validity::Array`], its length must exactly equal `offsets.len() - 1`.
    pub unsafe fn new_unchecked(
        offsets: ArrayRef,
        bytes: ByteBuffer,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        // SAFETY: `new_unchecked_from_handle` has same invariants which should be checked
        //  by caller.
        unsafe {
            Self::new_unchecked_from_handle(offsets, BufferHandle::new_host(bytes), dtype, validity)
        }
    }

    /// Creates a new [`VarBinArray`] without validation from its components, with string data
    /// stored in a `BufferHandle` (CPU or GPU).
    ///
    /// # Safety
    ///
    /// The caller must ensure all the invariants documented in `new_unchecked` are satisfied.
    pub unsafe fn new_unchecked_from_handle(
        offsets: ArrayRef,
        bytes: BufferHandle,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&offsets, &bytes, &dtype, &validity)
            .vortex_expect("[Debug Assertion]: Invalid `VarBinArray` parameters");

        let len = offsets.len().saturating_sub(1);
        let validity_slot = validity_to_child(&validity, len);

        Self {
            dtype,
            bytes,
            slots: vec![Some(offsets), validity_slot],
            validity,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`VarBinArray`].
    ///
    /// This function checks all the invariants required by [`VarBinArray::new_unchecked`].
    pub fn validate(
        offsets: &ArrayRef,
        bytes: &BufferHandle,
        dtype: &DType,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Check offsets are non-nullable integer
        vortex_ensure!(
            offsets.dtype().is_int() && !offsets.dtype().is_nullable(),
            MismatchedTypes: "non nullable int", offsets.dtype()
        );

        // Check dtype is Binary or Utf8
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            MismatchedTypes: "utf8 or binary", dtype
        );

        // Check nullability matches
        vortex_ensure!(
            dtype.is_nullable() != matches!(validity, Validity::NonNullable),
            InvalidArgument: "incorrect validity {:?} for dtype {}",
            validity,
            dtype
        );

        // Check offsets has at least one element
        vortex_ensure!(
            !offsets.is_empty(),
            InvalidArgument: "Offsets must have at least one element"
        );

        // Skip host-only validation when offsets/bytes are not host-resident.
        if offsets.is_host() && bytes.is_on_host() {
            let last_offset = offsets
                .scalar_at(offsets.len() - 1)?
                .as_primitive()
                .as_::<usize>()
                .ok_or_else(
                    || vortex_err!(InvalidArgument: "Last offset must be convertible to usize"),
                )?;
            vortex_ensure!(
                last_offset <= bytes.len(),
                InvalidArgument: "Last offset {} exceeds bytes length {}",
                last_offset,
                bytes.len()
            );
        }

        // Check validity length
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == offsets.len() - 1,
                "Validity length {} doesn't match array length {}",
                validity_len,
                offsets.len() - 1
            );
        }

        // Validate UTF-8 for Utf8 dtype. Skip when offsets/bytes are not host-resident.
        if offsets.is_host()
            && bytes.is_on_host()
            && matches!(dtype, DType::Utf8(_))
            && let Some(bytes) = bytes.as_host_opt()
        {
            let primitive_offsets = offsets.to_primitive();
            match_each_integer_ptype!(primitive_offsets.dtype().as_ptype(), |O| {
                let offsets_slice = primitive_offsets.as_slice::<O>();
                for (i, (start, end)) in offsets_slice
                    .windows(2)
                    .map(|o| (o[0].as_(), o[1].as_()))
                    .enumerate()
                {
                    if validity.is_null(i)? {
                        continue;
                    }

                    let string_bytes = &bytes.as_ref()[start..end];
                    simdutf8::basic::from_utf8(string_bytes).map_err(|_| {
                        #[allow(clippy::unwrap_used)]
                        // run validation using `compat` package to get more detailed error message
                        let err = simdutf8::compat::from_utf8(string_bytes).unwrap_err();
                        vortex_err!("invalid utf-8: {err} at index {i}")
                    })?;
                }
            });
        }

        Ok(())
    }

    #[inline]
    pub fn offsets(&self) -> &ArrayRef {
        self.slots[OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("VarBinArray offsets slot")
    }

    /// Access the value bytes child buffer
    ///
    /// # Note
    ///
    /// Bytes child buffer is never sliced when the array is sliced so this can include values
    /// that are not logically present in the array. Users should prefer [sliced_bytes][Self::sliced_bytes]
    /// unless they're resolving values via the offset child array.
    #[inline]
    pub fn bytes(&self) -> &ByteBuffer {
        self.bytes.as_host()
    }

    /// Access the value bytes buffer handle.
    #[inline]
    pub fn bytes_handle(&self) -> &BufferHandle {
        &self.bytes
    }

    /// Access value bytes child array limited to values that are logically present in
    /// the array unlike [bytes][Self::bytes].
    pub fn sliced_bytes(&self) -> ByteBuffer {
        let first_offset: usize = self.offset_at(0);
        let last_offset = self.offset_at(self.len());

        self.bytes().slice(first_offset..last_offset)
    }

    pub fn from_vec<T: AsRef<[u8]>>(vec: Vec<T>, dtype: DType) -> Self {
        let size: usize = vec.iter().map(|v| v.as_ref().len()).sum();
        if size < u32::MAX as usize {
            Self::from_vec_sized::<u32, T>(vec, dtype)
        } else {
            Self::from_vec_sized::<u64, T>(vec, dtype)
        }
    }

    fn from_vec_sized<O, T>(vec: Vec<T>, dtype: DType) -> Self
    where
        O: IntegerPType,
        T: AsRef<[u8]>,
    {
        let mut builder = VarBinBuilder::<O>::with_capacity(vec.len());
        for v in vec {
            builder.append_value(v.as_ref());
        }
        builder.finish(dtype)
    }

    #[expect(
        clippy::same_name_method,
        reason = "intentionally named from_iter like Iterator::from_iter"
    )]
    pub fn from_iter<T: AsRef<[u8]>, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinBuilder::<u32>::with_capacity(iter.size_hint().0);
        for v in iter {
            builder.append(v.as_ref().map(|o| o.as_ref()));
        }
        builder.finish(dtype)
    }

    pub fn from_iter_nonnull<T: AsRef<[u8]>, I: IntoIterator<Item = T>>(
        iter: I,
        dtype: DType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut builder = VarBinBuilder::<u32>::with_capacity(iter.size_hint().0);
        for v in iter {
            builder.append_value(v);
        }
        builder.finish(dtype)
    }

    /// Get value offset at a given index
    ///
    /// Note: There's 1 more offsets than the elements in the array, thus last offset is at array length index
    ///
    /// Panics if index is out of bounds
    pub fn offset_at(&self, index: usize) -> usize {
        assert!(
            index <= self.len(),
            "Index {index} out of bounds 0..={}",
            self.len()
        );

        (&self
            .offsets()
            .scalar_at(index)
            .vortex_expect("offsets must support scalar_at"))
            .try_into()
            .vortex_expect("Failed to convert offset to usize")
    }

    /// Access value bytes at a given index
    ///
    /// Will return buffer referencing underlying data without performing a copy
    pub fn bytes_at(&self, index: usize) -> ByteBuffer {
        let start = self.offset_at(index);
        let end = self.offset_at(index + 1);

        self.bytes().slice(start..end)
    }

    /// Consumes self, returning a tuple containing the `DType`, the `bytes` array,
    /// the `offsets` array, and the `validity`.
    pub fn into_parts(mut self) -> (DType, BufferHandle, ArrayRef, Validity) {
        let offsets = self.slots[OFFSETS_SLOT]
            .take()
            .vortex_expect("VarBinArray offsets slot");
        (self.dtype, self.bytes, offsets, self.validity)
    }
}

impl From<Vec<&[u8]>> for VarBinArray {
    fn from(value: Vec<&[u8]>) -> Self {
        Self::from_vec(value, DType::Binary(Nullability::NonNullable))
    }
}

impl From<Vec<Vec<u8>>> for VarBinArray {
    fn from(value: Vec<Vec<u8>>) -> Self {
        Self::from_vec(value, DType::Binary(Nullability::NonNullable))
    }
}

impl From<Vec<String>> for VarBinArray {
    fn from(value: Vec<String>) -> Self {
        Self::from_vec(value, DType::Utf8(Nullability::NonNullable))
    }
}

impl From<Vec<&str>> for VarBinArray {
    fn from(value: Vec<&str>) -> Self {
        Self::from_vec(value, DType::Utf8(Nullability::NonNullable))
    }
}

impl From<Vec<Option<&[u8]>>> for VarBinArray {
    fn from(value: Vec<Option<&[u8]>>) -> Self {
        Self::from_iter(value, DType::Binary(Nullability::Nullable))
    }
}

impl From<Vec<Option<Vec<u8>>>> for VarBinArray {
    fn from(value: Vec<Option<Vec<u8>>>) -> Self {
        Self::from_iter(value, DType::Binary(Nullability::Nullable))
    }
}

impl From<Vec<Option<String>>> for VarBinArray {
    fn from(value: Vec<Option<String>>) -> Self {
        Self::from_iter(value, DType::Utf8(Nullability::Nullable))
    }
}

impl From<Vec<Option<&str>>> for VarBinArray {
    fn from(value: Vec<Option<&str>>) -> Self {
        Self::from_iter(value, DType::Utf8(Nullability::Nullable))
    }
}

impl<'a> FromIterator<Option<&'a [u8]>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<&'a [u8]>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Binary(Nullability::Nullable))
    }
}

impl FromIterator<Option<Vec<u8>>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<Vec<u8>>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Binary(Nullability::Nullable))
    }
}

impl FromIterator<Option<String>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<String>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Utf8(Nullability::Nullable))
    }
}

impl<'a> FromIterator<Option<&'a str>> for VarBinArray {
    fn from_iter<T: IntoIterator<Item = Option<&'a str>>>(iter: T) -> Self {
        Self::from_iter(iter, DType::Utf8(Nullability::Nullable))
    }
}
