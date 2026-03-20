// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Terminal execution node wrapping a pre-built Arrow array.
//!
//! `NativeArrowArray` is a transient wrapper produced during Arrow export.
//! It participates in the execution loop as a terminal node (`execute()` returns `Done(self)`).
//! It is never serialized to disk or IPC.

use std::hash::Hasher;

use arrow_array::ArrayRef as ArrowArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::Precision;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

vtable!(NativeArrow);

#[derive(Debug)]
pub struct NativeArrow;

impl NativeArrow {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.native_arrow");
}

/// A transient array wrapping a pre-built Arrow array.
///
/// This is only used during Arrow export as a terminal node in the execution loop.
#[derive(Clone, Debug)]
pub struct NativeArrowArray {
    arrow: ArrowArrayRef,
    dtype: DType,
    stats_set: ArrayStats,
}

impl NativeArrowArray {
    /// Create a new `NativeArrowArray` wrapping the given Arrow array.
    pub fn new(arrow: ArrowArrayRef, dtype: DType) -> Self {
        Self {
            arrow,
            dtype,
            stats_set: Default::default(),
        }
    }

    /// Returns a reference to the underlying Arrow array.
    pub fn arrow_array(&self) -> &ArrowArrayRef {
        &self.arrow
    }
}

impl VTable for NativeArrow {
    type Array = NativeArrowArray;
    type Metadata = ();
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &NativeArrowArray) -> usize {
        array.arrow.len()
    }

    fn dtype(array: &NativeArrowArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &NativeArrowArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(_array: &NativeArrowArray, _state: &mut H, _precision: Precision) {
        vortex_panic!("NativeArrowArray is transient and does not support hashing")
    }

    fn array_eq(
        _array: &NativeArrowArray,
        _other: &NativeArrowArray,
        _precision: Precision,
    ) -> bool {
        vortex_panic!("NativeArrowArray is transient and does not support equality")
    }

    fn nbuffers(_array: &NativeArrowArray) -> usize {
        0
    }

    fn buffer(_array: &NativeArrowArray, idx: usize) -> BufferHandle {
        vortex_panic!("NativeArrowArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &NativeArrowArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &NativeArrowArray) -> usize {
        0
    }

    fn child(_array: &NativeArrowArray, idx: usize) -> ArrayRef {
        vortex_panic!("NativeArrowArray child index {idx} out of bounds")
    }

    fn child_name(_array: &NativeArrowArray, idx: usize) -> String {
        vortex_panic!("NativeArrowArray child_name index {idx} out of bounds")
    }

    fn metadata(_array: &NativeArrowArray) -> VortexResult<Self::Metadata> {
        Ok(())
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // Transient — never serialized.
        Ok(None)
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_bail!("NativeArrowArray cannot be deserialized")
    }

    fn build(
        _dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<NativeArrowArray> {
        vortex_bail!("NativeArrowArray cannot be built from components")
    }

    fn with_children(_array: &mut Self::Array, _children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_bail!("NativeArrowArray has no children")
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(array.clone().into_array()))
    }
}
