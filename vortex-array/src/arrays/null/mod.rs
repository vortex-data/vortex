// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::null::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;

const NUM_SLOTS: usize = 0;

pub(crate) mod compute;

vtable!(Null, Null, NullData);

impl VTable for Null {
    type ArrayData = NullData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, _data: &NullData, dtype: &DType, _len: usize) -> VortexResult<()> {
        vortex_ensure!(*dtype == DType::Null, "NullArray dtype must be DType::Null");
        Ok(())
    }

    fn array_hash<H: std::hash::Hasher>(_array: &NullData, _state: &mut H, _precision: Precision) {
        // len and dtype are hashed by ArrayInner; NullData has no additional fields.
    }

    fn array_eq(_array: &NullData, _other: &NullData, _precision: Precision) -> bool {
        // len and dtype are compared by ArrayInner; NullData has no additional fields.
        true
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("NullArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("NullArray slot_name index {idx} out of bounds")
    }

    fn with_slots(_array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "NullArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        Ok(())
    }

    fn serialize(_array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<NullData> {
        <EmptyMetadata as crate::DeserializeMetadata>::deserialize(metadata)?;
        Ok(NullData::new())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}

/// A array where all values are null.
///
/// This mirrors the Apache Arrow Null array encoding and provides an efficient representation
/// for arrays containing only null values. No actual data is stored, only the length.
///
/// All operations on null arrays return null values or indicate invalid data.
///
/// # Examples
///
/// ```
/// # fn main() -> vortex_error::VortexResult<()> {
/// use vortex_array::arrays::NullArray;
/// use vortex_array::IntoArray;
///
/// // Create a null array with 5 elements
/// let array = NullArray::new(5);
///
/// // Slice the array - still contains nulls
/// let sliced = array.slice(1..3)?;
/// assert_eq!(sliced.len(), 2);
///
/// // All elements are null
/// let scalar = array.scalar_at(0).unwrap();
/// assert!(scalar.is_null());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct NullData {
    slots: Vec<Option<ArrayRef>>,
}

#[derive(Clone, Debug)]
pub struct Null;

impl Null {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.null");
}

impl Array<Null> {
    pub fn new(len: usize) -> Self {
        Array::try_from_parts(ArrayParts::new(Null, DType::Null, len, NullData::new()))
            .vortex_expect("NullData is always valid")
    }
}

impl NullData {
    pub fn new() -> Self {
        Self { slots: vec![] }
    }
}
impl OperationsVTable<Null> for Null {
    fn scalar_at(
        _array: ArrayView<'_, Null>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::null(DType::Null))
    }
}

impl ValidityVTable<Null> for Null {
    fn validity(_array: ArrayView<'_, Null>) -> VortexResult<Validity> {
        Ok(Validity::AllInvalid)
    }
}
