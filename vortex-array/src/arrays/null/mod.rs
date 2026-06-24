// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::null::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

pub(crate) mod compute;

/// A [`Null`]-encoded Vortex array.
pub type NullArray = Array<Null>;

impl VTable for Null {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.null");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(*dtype == DType::Null, "NullArray dtype must be DType::Null");
        Ok(())
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("NullArray slot_name index {idx} out of bounds")
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

        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            metadata.is_empty(),
            "NullArray expects empty metadata, got {} bytes",
            metadata.len()
        );
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            EmptyArrayData,
        ))
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
/// use vortex_array::{IntoArray, VortexSessionExecute, array_session};
///
/// // Create a null array with 5 elements
/// let array = NullArray::new(5);
///
/// // Slice the array - still contains nulls
/// let sliced = array.slice(1..3)?;
/// assert_eq!(sliced.len(), 2);
///
/// // All elements are null
/// let mut ctx = array_session().create_execution_ctx();
/// let scalar = array.execute_scalar(0, &mut ctx).unwrap();
/// assert!(scalar.is_null());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Null;

impl Array<Null> {
    pub fn new(len: usize) -> Self {
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Null, DType::Null, len, EmptyArrayData))
        }
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
