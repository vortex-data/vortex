// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

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
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

pub(crate) mod compute;

vtable!(Null);

impl VTable for Null {
    type Array = NullArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &Null
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &NullArray) -> usize {
        array.len
    }

    fn dtype(_array: &NullArray) -> &DType {
        &DType::Null
    }

    fn stats(array: &NullArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &NullArray, state: &mut H, _precision: Precision) {
        array.len.hash(state);
    }

    fn array_eq(array: &NullArray, other: &NullArray, _precision: Precision) -> bool {
        array.len == other.len
    }

    fn nbuffers(_array: &NullArray) -> usize {
        0
    }

    fn buffer(_array: &NullArray, idx: usize) -> BufferHandle {
        vortex_panic!("NullArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &NullArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &NullArray) -> usize {
        0
    }

    fn child(_array: &NullArray, idx: usize) -> ArrayRef {
        vortex_panic!("NullArray child index {idx} out of bounds")
    }

    fn child_name(_array: &NullArray, idx: usize) -> String {
        vortex_panic!("NullArray child_name index {idx} out of bounds")
    }

    fn metadata(_array: &NullArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<NullArray> {
        Ok(NullArray::new(len))
    }

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "NullArray has no children, got {}",
            children.len()
        );
        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Arc<Self::Array>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done_upcast::<Self>(array))
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
pub struct NullArray {
    len: usize,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct Null;

impl Null {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.null");
}

impl NullArray {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            stats_set: Default::default(),
        }
    }
}
impl OperationsVTable<Null> for Null {
    fn scalar_at(
        _array: &NullArray,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::null(DType::Null))
    }
}

impl ValidityVTable<Null> for Null {
    fn validity(_array: &NullArray) -> VortexResult<Validity> {
        Ok(Validity::AllInvalid)
    }
}
