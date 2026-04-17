// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::BoolArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::kernel::PARENT_KERNELS;

/// A [`ByteBool`]-encoded Vortex array.
pub type ByteBoolArray = Array<ByteBool>;

impl ArrayHash for ByteBoolData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.buffer.array_hash(state, precision);
    }
}

impl ArrayEq for ByteBoolData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.buffer.array_eq(&other.buffer, precision)
    }
}

impl VTable for ByteBool {
    type ArrayData = ByteBoolData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.bytebool");
        *ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity = child_to_validity(&slots[VALIDITY_SLOT], dtype.nullability());
        ByteBoolData::validate(data.buffer(), &validity, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.buffer().clone(),
            _ => vortex_panic!("ByteBoolArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("values".to_string()),
            _ => vortex_panic!("ByteBoolArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        if !metadata.is_empty() {
            vortex_bail!(
                "ByteBoolArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        let data = ByteBoolData::try_new(buffer, validity.clone())?;
        let slots = ByteBoolData::make_slots(&validity, len);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let boolean_buffer = BitBuffer::from(array.as_slice());
        let validity = array.validity()?;
        Ok(ExecutionResult::done(
            BoolArray::new(boolean_buffer, validity).into_array(),
        ))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

/// The validity bitmap indicating which elements are non-null.
pub(super) const VALIDITY_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

#[derive(Clone, Debug)]
pub struct ByteBoolData {
    buffer: BufferHandle,
}

impl Display for ByteBoolData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait ByteBoolArrayExt: TypedArrayRef<ByteBool> {
    fn validity(&self) -> Validity {
        child_to_validity(
            &self.as_ref().slots()[VALIDITY_SLOT],
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: TypedArrayRef<ByteBool>> ByteBoolArrayExt for T {}

#[derive(Clone, Debug)]
pub struct ByteBool;

impl ByteBool {
    /// Construct a [`ByteBoolArray`] from a raw bytes buffer and validity.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented in
    /// [`ByteBool::new_unchecked`].
    pub fn new(buffer: BufferHandle, validity: Validity) -> ByteBoolArray {
        Self::try_new(buffer, validity).vortex_expect("ByteBoolArray construction failed")
    }

    /// Construct a [`ByteBoolArray`] from a raw bytes buffer and validity, returning
    /// an error if the provided components do not satisfy the invariants documented
    /// in [`ByteBool::new_unchecked`].
    pub fn try_new(buffer: BufferHandle, validity: Validity) -> VortexResult<ByteBoolArray> {
        let dtype = DType::Bool(validity.nullability());
        let len = buffer.len();
        let slots = ByteBoolData::make_slots(&validity, len);
        let data = ByteBoolData::try_new(buffer, validity)?;
        Array::try_from_parts(ArrayParts::new(ByteBool, dtype, len, data).with_slots(slots))
    }

    /// Construct a [`ByteBoolArray`] without validating the buffer contents.
    ///
    /// # Safety
    ///
    /// Every byte of `buffer` must be `0x00` or `0x01`. Any other byte value is
    /// Undefined Behavior because [`ByteBoolData::as_slice`] reinterprets the buffer
    /// as `&[bool]`, and a `bool` with any bit pattern other than 0 or 1 is UB.
    /// If `validity` is [`Validity::Array`], its length must equal `buffer.len()`.
    pub unsafe fn new_unchecked(buffer: BufferHandle, validity: Validity) -> ByteBoolArray {
        let dtype = DType::Bool(validity.nullability());
        let len = buffer.len();
        let slots = ByteBoolData::make_slots(&validity, len);
        // SAFETY: caller guarantees every buffer byte is 0 or 1.
        let data = unsafe { ByteBoolData::new_unchecked(buffer, validity) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(ByteBool, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Construct a [`ByteBoolArray`] from a `Vec<bool>` and validity.
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> ByteBoolArray {
        let validity = validity.into();
        let data = ByteBoolData::from_vec(data, validity.clone());
        let dtype = DType::Bool(validity.nullability());
        let len = data.len();
        let slots = ByteBoolData::make_slots(&validity, len);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(ByteBool, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Construct a [`ByteBoolArray`] from optional bools.
    pub fn from_option_vec(data: Vec<Option<bool>>) -> ByteBoolArray {
        let validity = Validity::from_iter(data.iter().map(|v| v.is_some()));
        let data = ByteBoolData::from(data);
        let dtype = DType::Bool(validity.nullability());
        let len = data.len();
        let slots = ByteBoolData::make_slots(&validity, len);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(ByteBool, dtype, len, data).with_slots(slots),
            )
        }
    }
}

impl ByteBoolData {
    pub fn validate(
        buffer: &BufferHandle,
        validity: &Validity,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        let expected_dtype = DType::Bool(validity.nullability());
        vortex_ensure!(
            dtype == &expected_dtype,
            "expected dtype {expected_dtype}, got {dtype}"
        );
        vortex_ensure!(
            buffer.len() == len,
            "expected len {len}, got {}",
            buffer.len()
        );
        if let Some(vlen) = validity.maybe_len() {
            vortex_ensure!(vlen == len, "expected validity len {len}, got {vlen}");
        }
        Ok(())
    }

    /// Validate that every byte of `buffer` is `0x00` or `0x01`.
    ///
    /// [`ByteBoolData::as_slice`] transmutes the buffer's bytes to `&[bool]`; any byte
    /// other than `0x00` or `0x01` would produce a `bool` with an invalid bit pattern,
    /// which is Undefined Behavior per the Rust reference.
    ///
    /// Device-resident buffers are not host-readable and are skipped.
    pub fn validate_bytes(buffer: &BufferHandle) -> VortexResult<()> {
        let Some(bytes) = buffer.as_host_opt() else {
            return Ok(());
        };
        // Count over a flat `&[u8]` vectorizes to pcmpgtb/pmovmskb on x86 and
        // cmhi/addv on aarch64, so the fast path runs at ~16 bytes/cycle.
        // See https://godbolt.org/z/z797nT1c8
        let invalid = bytes.as_slice().iter().filter(|&&b| b > 1).count();
        vortex_ensure_eq!(
            invalid,
            0,
            "ByteBoolArray buffer contains {invalid} bytes that are not 0 or 1",
        );
        Ok(())
    }

    fn make_slots(validity: &Validity, len: usize) -> Vec<Option<ArrayRef>> {
        vec![validity_to_child(validity, len)]
    }

    /// Construct [`ByteBoolData`] from a raw bytes buffer and validity.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented in
    /// [`ByteBoolData::new_unchecked`].
    pub fn new(buffer: BufferHandle, validity: Validity) -> Self {
        Self::try_new(buffer, validity).vortex_expect("ByteBoolData construction failed")
    }

    /// Construct [`ByteBoolData`] from a raw bytes buffer and validity, returning an
    /// error if the provided components do not satisfy the invariants documented in
    /// [`ByteBoolData::new_unchecked`].
    pub fn try_new(buffer: BufferHandle, validity: Validity) -> VortexResult<Self> {
        Self::check_validity_len(&buffer, &validity)?;
        Self::validate_bytes(&buffer)?;
        // SAFETY: buffer bytes and validity length validated above.
        Ok(unsafe { Self::new_unchecked(buffer, validity) })
    }

    /// Construct [`ByteBoolData`] without validating the buffer contents.
    ///
    /// # Safety
    ///
    /// Every byte of `buffer` must be `0x00` or `0x01`. Any other byte value is
    /// Undefined Behavior because [`ByteBoolData::as_slice`] reinterprets the buffer
    /// as `&[bool]`, and a `bool` with any bit pattern other than 0 or 1 is UB.
    /// If `validity` is [`Validity::Array`], its length must equal `buffer.len()`.
    pub unsafe fn new_unchecked(buffer: BufferHandle, validity: Validity) -> Self {
        debug_assert!(
            Self::validate_bytes(&buffer).is_ok(),
            "ByteBoolData::new_unchecked called with non-boolean bytes",
        );
        Self::check_validity_len(&buffer, &validity)
            .vortex_expect("ByteBoolData::new_unchecked called with mismatched validity length");
        Self { buffer }
    }

    fn check_validity_len(buffer: &BufferHandle, validity: &Validity) -> VortexResult<()> {
        if let Some(vlen) = validity.maybe_len() {
            vortex_ensure!(
                buffer.len() == vlen,
                "Buffer length ({}) does not match validity length ({})",
                buffer.len(),
                vlen
            );
        }
        Ok(())
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.len() == 0
    }

    // TODO(ngates): deprecate construction from vec
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> Self {
        let validity = validity.into();
        // SAFETY: we are transmuting a Vec<bool> into a Vec<u8>
        let data: Vec<u8> = unsafe { std::mem::transmute(data) };
        let buffer = BufferHandle::new_host(ByteBuffer::from(data));
        // SAFETY: bytes came from `Vec<bool>`, which guarantees values of 0 or 1.
        unsafe { Self::new_unchecked(buffer, validity) }
    }

    pub fn buffer(&self) -> &BufferHandle {
        &self.buffer
    }

    pub fn as_slice(&self) -> &[bool] {
        // Safety: The internal buffer contains byte-sized bools
        unsafe { std::mem::transmute(self.buffer().as_host().as_slice()) }
    }
}

impl ValidityVTable<ByteBool> for ByteBool {
    fn validity(array: ArrayView<'_, ByteBool>) -> VortexResult<Validity> {
        Ok(ByteBoolArrayExt::validity(&array))
    }
}

impl OperationsVTable<ByteBool> for ByteBool {
    fn scalar_at(
        array: ArrayView<'_, ByteBool>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.buffer.as_host()[index] == 1,
            array.dtype().nullability(),
        ))
    }
}

impl From<Vec<bool>> for ByteBoolData {
    fn from(value: Vec<bool>) -> Self {
        Self::from_vec(value, Validity::AllValid)
    }
}

impl From<Vec<Option<bool>>> for ByteBoolData {
    fn from(value: Vec<Option<bool>>) -> Self {
        let validity = Validity::from_iter(value.iter().map(|v| v.is_some()));

        // This doesn't reallocate, and the compiler even vectorizes it
        let data = value.into_iter().map(Option::unwrap_or_default).collect();

        Self::from_vec(data, validity)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySession;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = ByteBool::from_vec(v, Validity::AllValid);
        assert_eq!(v_len, arr.len());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx, &mut ctx).unwrap());
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBool::from_option_vec(v);
        assert!(arr.is_valid(0, &mut ctx).unwrap());
        assert!(!arr.is_valid(1, &mut ctx).unwrap());
        assert!(arr.is_valid(2, &mut ctx).unwrap());
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBool::from_option_vec(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx, &mut ctx).unwrap());
        }
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn try_new_rejects_invalid_byte() {
        // `ByteBoolData::as_slice` transmutes the underlying bytes into `&[bool]`.
        // A bool with any bit pattern other than 0 or 1 is Undefined Behavior per
        // the Rust reference, so `try_new` must reject these buffers.
        let raw = ByteBuffer::from(vec![0x02u8, 0x01, 0xFFu8]);
        let handle = BufferHandle::new_host(raw);
        let err = ByteBool::try_new(handle, Validity::NonNullable).unwrap_err();
        assert!(
            err.to_string().contains("bytes that are not 0 or 1"),
            "unexpected error: {err}",
        );
    }

    #[test]
    #[should_panic(expected = "bytes that are not 0 or 1")]
    fn new_panics_on_invalid_byte() {
        let raw = ByteBuffer::from(vec![0x02u8]);
        let handle = BufferHandle::new_host(raw);
        drop(ByteBool::new(handle, Validity::NonNullable));
    }

    #[test]
    fn new_unchecked_accepts_valid_bytes() {
        let raw = ByteBuffer::from(vec![0x00u8, 0x01, 0x01, 0x00]);
        let handle = BufferHandle::new_host(raw);
        // SAFETY: all bytes are 0 or 1.
        let arr = unsafe { ByteBool::new_unchecked(handle, Validity::NonNullable) };
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.as_slice(), &[false, true, true, false]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-boolean bytes")]
    fn new_unchecked_debug_asserts_invalid_bytes() {
        let raw = ByteBuffer::from(vec![0x02u8]);
        let handle = BufferHandle::new_host(raw);
        // SAFETY: intentionally violated to exercise the debug assertion.
        drop(unsafe { ByteBool::new_unchecked(handle, Validity::NonNullable) });
    }

    #[test]
    fn test_nullable_bytebool_serde_roundtrip() {
        let array = ByteBool::from_option_vec(vec![Some(true), None, Some(false), None]);
        let dtype = array.dtype().clone();
        let len = array.len();
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(ByteBool);

        let ctx = ArrayContext::empty();
        let serialized = array
            .clone()
            .into_array()
            .serialize(&ctx, &session, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }

        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
            .unwrap();

        assert_arrays_eq!(decoded, array);
    }
}
