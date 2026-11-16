// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;
use std::iter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::repeated::RepeatedArraySlotsExt;
use crate::arrays::repeated::RepeatedData;
use crate::arrays::repeated::RepeatedSlots;
use crate::arrays::repeated::array::repeated_len;
use crate::arrays::repeated::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// The encoding of a [`RepeatedArray`].
#[derive(Clone, Debug)]
pub struct Repeated;

/// A [`Repeated`]-encoded array, which repeats its child end to end.
pub type RepeatedArray = Array<Repeated>;

impl ArrayHash for RepeatedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.count.hash(state);
    }
}

impl ArrayEq for RepeatedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.count == other.count
    }
}

impl VTable for Repeated {
    type TypedArrayData = RepeatedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.repeated");
        *ID
    }

    fn validate(
        &self,
        data: &RepeatedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), RepeatedSlots::COUNT);
        let child = slots[RepeatedSlots::CHILD]
            .as_ref()
            .ok_or_else(|| vortex_err!("Repeated requires a child"))?;
        vortex_ensure_eq!(
            child.dtype(),
            dtype,
            "Repeated child dtype does not match outer dtype"
        );
        vortex_ensure_eq!(
            repeated_len(child.len(), data.count)?,
            len,
            "Repeated length does not match child length times count"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Repeated buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
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
        vortex_bail!("Repeated array is not serializable")
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if array.is_empty() {
            return Ok(());
        }
        let child = array
            .child()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
        for _ in 0..array.count {
            child.append_to_builder(builder, ctx)?;
        }
        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        RepeatedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if array.child().len() == 1 {
            let scalar = array.child().execute_scalar(0, ctx)?;
            return Ok(ExecutionResult::done(
                ConstantArray::new(scalar, array.len()).into_array(),
            ));
        }

        let array = require_child!(array, array.child(), RepeatedSlots::CHILD => AnyCanonical);

        // Decode the child once, then let `Chunked` append the decoded child once per copy.
        let copies = iter::repeat_n(array.child().clone(), array.count);
        let chunked = ChunkedArray::try_new(copies, array.dtype().clone())?;
        Ok(ExecutionResult::done(chunked.into_array()))
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        if array.is_empty() {
            return Ok(Some(Canonical::empty(array.dtype()).into_array()));
        }
        if array.count == 1 {
            return Ok(Some(array.child().clone()));
        }
        if let Some(constant) = array.child().as_opt::<Constant>() {
            return Ok(Some(
                ConstantArray::new(constant.scalar().clone(), array.len()).into_array(),
            ));
        }
        if let Some(inner) = array.child().as_opt::<Repeated>() {
            let count = inner
                .count
                .checked_mul(array.count)
                .ok_or_else(|| vortex_err!("Repeated count overflows usize"))?;
            return Ok(Some(
                RepeatedArray::try_new(inner.child().clone(), count)?.into_array(),
            ));
        }
        Ok(None)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<Repeated> for Repeated {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Repeated>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let child = state
            .slot(RepeatedSlots::CHILD)?
            .ok_or_else(|| vortex_err!("Repeated child slot is missing"))?;
        let child_index = index % child.len();
        child.execute_scalar(child_index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Repeated>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl ValidityVTable<Repeated> for Repeated {
    fn validity(array: ArrayView<'_, Repeated>) -> VortexResult<Validity> {
        Ok(match array.child().validity()? {
            Validity::Array(validity) => {
                Validity::Array(RepeatedArray::try_new(validity, array.count)?.into_array())
            }
            validity => validity,
        })
    }
}
