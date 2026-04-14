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
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
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

        let data = ByteBoolData::new(buffer, validity.clone());
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

    fn validity_mask(&self) -> Mask {
        self.validity().to_mask(self.as_ref().len())
    }
}

impl<T: TypedArrayRef<ByteBool>> ByteBoolArrayExt for T {}

#[derive(Clone, Debug)]
pub struct ByteBool;

impl ByteBool {
    pub fn new(buffer: BufferHandle, validity: Validity) -> ByteBoolArray {
        let dtype = DType::Bool(validity.nullability());
        let slots = ByteBoolData::make_slots(&validity, buffer.len());
        let data = ByteBoolData::new(buffer, validity);
        let len = data.len();
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

    fn make_slots(validity: &Validity, len: usize) -> Vec<Option<ArrayRef>> {
        vec![validity_to_child(validity, len)]
    }

    pub fn new(buffer: BufferHandle, validity: Validity) -> Self {
        let length = buffer.len();
        if let Some(vlen) = validity.maybe_len()
            && length != vlen
        {
            vortex_panic!(
                "Buffer length ({}) does not match validity length ({})",
                length,
                vlen
            );
        }
        Self { buffer }
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
        Self::new(BufferHandle::new_host(ByteBuffer::from(data)), validity)
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

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx).unwrap());
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBool::from_option_vec(v);
        assert!(arr.is_valid(0).unwrap());
        assert!(!arr.is_valid(1).unwrap());
        assert!(arr.is_valid(2).unwrap());
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBool::from_option_vec(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx).unwrap());
        }
        assert_eq!(arr.len(), 2);
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
            .serialize(&ctx, &LEGACY_SESSION, &SerializeOptions::default())
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
