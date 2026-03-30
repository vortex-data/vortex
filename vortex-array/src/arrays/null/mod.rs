// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

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
use crate::arrays::null::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

pub(crate) mod compute;

vtable!(Null, Null, NullData);

impl VTable for Null {
    type ArrayData = NullData;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Null
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &NullData) -> usize {
        array.len
    }

    fn dtype(_array: &NullData) -> &DType {
        &DType::Null
    }

    fn stats(array: &NullData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &Array<Self>, state: &mut H, _precision: Precision) {
        array.len.hash(state);
    }

    fn array_eq(array: &Array<Self>, other: &Array<Self>, _precision: Precision) -> bool {
        array.len == other.len
    }

    fn nbuffers(_array: &Array<Self>) -> usize {
        0
    }

    fn buffer(_array: &Array<Self>, idx: usize) -> BufferHandle {
        vortex_panic!("NullArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &Array<Self>, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &Array<Self>) -> usize {
        0
    }

    fn child(_array: &Array<Self>, idx: usize) -> ArrayRef {
        vortex_panic!("NullArray child index {idx} out of bounds")
    }

    fn child_name(_array: &Array<Self>, idx: usize) -> String {
        vortex_panic!("NullArray child_name index {idx} out of bounds")
    }

    fn metadata(_array: &Array<Self>) -> VortexResult<Self::Metadata> {
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
        _dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<NullData> {
        Ok(NullData::new(len))
    }

    fn with_children(_array: &mut Self::ArrayData, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "NullArray has no children, got {}",
            children.len()
        );
        Ok(())
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
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
    len: usize,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct Null;

impl Null {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.null");
}

impl Array<Null> {
    pub fn new(len: usize) -> Self {
        Array::try_from_data(NullData::new(len)).vortex_expect("NullData is always valid")
    }
}

impl NullData {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            stats_set: Default::default(),
        }
    }

    /// Returns the dtype of the array (always [`DType::Null`]).
    pub fn dtype(&self) -> &DType {
        &DType::Null
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl OperationsVTable<Null> for Null {
    fn scalar_at(
        _array: &Array<Null>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::null(DType::Null))
    }
}

impl ValidityVTable<Null> for Null {
    fn validity(_array: &Array<Null>) -> VortexResult<Validity> {
        Ok(Validity::AllInvalid)
    }
}
