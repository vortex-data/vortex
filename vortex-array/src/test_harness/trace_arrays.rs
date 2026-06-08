// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small encodings used by trace snapshot tests.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VTable;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::vtable::NotSupported;
use crate::array::vtable::ValidityVTable;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::matcher::Matcher;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// Create a `StackParent(StackChild)` fixture.
///
/// `StackParent` requests `ExecuteSlot` until its child is `Primitive`. `StackChild` has one
/// declined parent kernel followed by one successful parent kernel, so strict trace snapshots can
/// assert that stack parent kernels run before the child decodes itself.
pub fn stack_parent_fixture() -> VortexResult<ArrayRef> {
    stack_parent(stack_child()?)
}

/// Create the child encoding used by [`stack_parent_fixture`].
pub fn stack_child() -> VortexResult<ArrayRef> {
    Ok(
        Array::try_from_parts(ArrayParts::new(StackChild, test_dtype(), 3, StackChildData))?
            .into_array(),
    )
}

/// Wrap `child` in the parent encoding used by [`stack_parent_fixture`].
pub fn stack_parent(child: ArrayRef) -> VortexResult<ArrayRef> {
    Ok(Array::try_from_parts(
        ArrayParts::new(
            StackParent,
            child.dtype().clone(),
            child.len(),
            StackParentData,
        )
        .with_slots(smallvec![Some(child)]),
    )?
    .into_array())
}

fn test_dtype() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

#[derive(Clone, Debug)]
struct StackParent;

#[derive(Clone, Debug)]
struct StackParentData;

impl Display for StackParentData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("stack-parent")
    }
}

impl ArrayHash for StackParentData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _eq_mode: EqMode) {}
}

impl ArrayEq for StackParentData {
    fn array_eq(&self, _other: &Self, _eq_mode: EqMode) -> bool {
        true
    }
}

impl ValidityVTable<StackParent> for StackParent {
    fn validity(_array: ArrayView<'_, StackParent>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for StackParent {
    type TypedArrayData = StackParentData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.stack-parent");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype == &test_dtype(), "unexpected stack parent dtype");
        vortex_ensure!(len == 3, "unexpected stack parent length");
        vortex_ensure!(slots.len() == 1, "stack parent must have one child slot");
        let Some(child) = &slots[0] else {
            vortex_bail!("stack parent child slot is missing");
        };
        vortex_ensure!(child.dtype() == dtype, "stack parent child dtype mismatch");
        vortex_ensure!(child.len() == len, "stack parent child length mismatch");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("StackParent buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("StackParent cannot be deserialized")
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("StackParent slot index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let Some(child) = array.slots()[0].as_ref() else {
            vortex_bail!("stack parent child slot is missing");
        };
        if !child.is::<Primitive>() {
            return Ok(ExecutionResult::execute_slot::<Primitive>(array, 0));
        }

        Ok(ExecutionResult::done(child.clone()))
    }
}

#[derive(Clone, Debug)]
struct StackChild;

#[derive(Clone, Debug)]
struct StackChildData;

impl Display for StackChildData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("stack-child")
    }
}

impl ArrayHash for StackChildData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _eq_mode: EqMode) {}
}

impl ArrayEq for StackChildData {
    fn array_eq(&self, _other: &Self, _eq_mode: EqMode) -> bool {
        true
    }
}

impl ValidityVTable<StackChild> for StackChild {
    fn validity(_array: ArrayView<'_, StackChild>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for StackChild {
    type TypedArrayData = StackChildData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.stack-child");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype == &test_dtype(), "unexpected stack child dtype");
        vortex_ensure!(len == 3, "unexpected stack child length");
        vortex_ensure!(slots.is_empty(), "stack child must not have slots");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("StackChild buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("StackChild cannot be deserialized")
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("StackChild slot index {idx} out of bounds")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        debug_assert!(array.slots().is_empty());
        Ok(ExecutionResult::done(PrimitiveArray::from_iter([
            99i32, 99, 99,
        ])))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        STACK_CHILD_PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

const STACK_CHILD_PARENT_KERNELS: ParentKernelSet<StackChild> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&StackDeclineKernel),
    ParentKernelSet::lift(&StackParentKernel),
]);

#[derive(Debug)]
struct StackDeclineKernel;

impl ExecuteParentKernel<StackChild> for StackDeclineKernel {
    type Parent = StackParent;

    fn execute_parent(
        &self,
        _array: ArrayView<'_, StackChild>,
        _parent: <Self::Parent as Matcher>::Match<'_>,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }
}

#[derive(Debug)]
struct StackParentKernel;

impl ExecuteParentKernel<StackChild> for StackParentKernel {
    type Parent = StackParent;

    fn execute_parent(
        &self,
        _array: ArrayView<'_, StackChild>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent
            .slots()
            .get(child_idx)
            .is_some_and(|slot| slot.is_none())
        {
            return Ok(Some(PrimitiveArray::from_iter([1i32, 2, 3]).into_array()));
        }

        Ok(None)
    }
}
