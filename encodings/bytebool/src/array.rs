// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::BoolArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::ValidityVTableFromValidityHelper;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::kernel::PARENT_KERNELS;

vtable!(ByteBool, ByteBool, ByteBoolData);

impl VTable for ByteBool {
    type ArrayData = ByteBoolData;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &ByteBool
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ByteBoolData) -> usize {
        array.buffer.len()
    }

    fn dtype(array: &ByteBoolData) -> &DType {
        &array.dtype
    }

    fn stats(array: &ByteBoolData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &ByteBoolData, state: &mut H, precision: Precision) {
        array.buffer.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &ByteBoolData, other: &ByteBoolData, precision: Precision) -> bool {
        array.buffer.array_eq(&other.buffer, precision)
            && array.validity.array_eq(&other.validity, precision)
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

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ByteBoolData> {
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

        Ok(ByteBoolData::new(buffer, validity))
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ByteBoolArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.validity = match &slots[VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.dtype.nullability()),
        };
        array.slots = slots;
        Ok(())
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
        let validity = array.validity().clone();
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
    dtype: DType,
    buffer: BufferHandle,
    validity: Validity,
    pub(super) slots: Vec<Option<ArrayRef>>,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ByteBool;

impl ByteBool {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.bytebool");

    /// Construct a [`ByteBoolArray`] from a `Vec<bool>` and validity.
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> ByteBoolArray {
        Array::try_from_data(ByteBoolData::from_vec(data, validity))
            .vortex_expect("ByteBoolData is always valid")
    }
}

impl ByteBoolData {
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
        let slots = Self::make_slots(&validity, length);
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            slots,
            stats_set: Default::default(),
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.len() == 0
    }

    /// Returns the logical data type of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the validity mask for this array.
    pub fn validity_mask(&self) -> Mask {
        self.validity.to_mask(self.len())
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

impl ValidityHelper for ByteBoolData {
    fn validity(&self) -> &Validity {
        &self.validity
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
    use vortex_array::assert_arrays_eq;
    use vortex_array::serde::ArrayParts;
    use vortex_array::serde::SerializeOptions;
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

        let arr = ByteBoolArray::try_from_data(ByteBoolData::from(v))
            .vortex_expect("ByteBoolData is always valid");
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx).unwrap());
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBoolArray::try_from_data(ByteBoolData::from(v))
            .vortex_expect("ByteBoolData is always valid");
        assert!(arr.is_valid(0).unwrap());
        assert!(!arr.is_valid(1).unwrap());
        assert!(arr.is_valid(2).unwrap());
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBoolArray::try_from_data(ByteBoolData::from(v))
            .vortex_expect("ByteBoolData is always valid");
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx).unwrap());
        }
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_nullable_bytebool_serde_roundtrip() {
        let array = ByteBoolArray::try_from_data(ByteBoolData::from(vec![
            Some(true),
            None,
            Some(false),
            None,
        ]))
        .unwrap();
        let dtype = array.dtype().clone();
        let len = array.len();
        let session = VortexSession::empty().with::<ArraySession>();
        session.arrays().register(ByteBool);

        let ctx = ArrayContext::empty();
        let serialized = array
            .clone()
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }

        let parts = ArrayParts::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
            .unwrap();

        assert_arrays_eq!(decoded, array);
    }
}
