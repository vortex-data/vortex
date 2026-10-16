// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::array_slots;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

#[array_slots(ScalarFnValidity)]
struct ValiditySlots {
    #[slot(0)]
    child: ArrayRef,
}

pub(crate) type ValidityArray = Array<ScalarFnValidity>;

/// Special case for scalar functions whose validity is Irreducible.
/// Does a single execute step of a child.
#[derive(Clone, Debug)]
pub(crate) struct ScalarFnValidity;

impl Array<ScalarFnValidity> {
    pub(crate) fn new(child: ArrayRef) -> Self {
        let len = child.len();
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                ScalarFnValidity,
                Validity::DTYPE,
                len,
                EmptyArrayData,
                ValiditySlots { child }.into_slots(),
            ))
        }
    }
}

impl VTable for ScalarFnValidity {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.scalar_fn_validity");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        _dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        // We construct it internally, valid by definition
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("ScalarFnValidity has no buffers")
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
        vortex_bail!("ScalarFnValidity deserialize not supported");
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ValiditySlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let len = array.len();
        let child = array.child().clone().execute::<ArrayRef>(ctx)?;
        Ok(ExecutionResult::done(child.validity()?.to_array(len)))
    }
}

impl OperationsVTable<ScalarFnValidity> for ScalarFnValidity {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, ScalarFnValidity>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let value = state
            .slot(ValiditySlots::CHILD)?
            .vortex_expect("ScalarFnValidity child slot is present")
            .execute_scalar(index, ctx)?;
        Ok(Scalar::bool(!value.is_null(), Nullability::NonNullable))
    }

    fn scalar_at(
        array: ArrayView<'_, ScalarFnValidity>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl ValidityVTable<ScalarFnValidity> for ScalarFnValidity {
    fn validity(_array: ArrayView<'_, ScalarFnValidity>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::consistency::test_scalar_probe_consistency;

    use super::ValidityArray;

    #[test]
    fn scalar_probe_child_nullness() -> VortexResult<()> {
        let child = PrimitiveArray::from_option_iter([
            Some(1i32),
            None,
            Some(3),
            None,
            Some(5),
        ])
        .into_array();
        let array = ValidityArray::new(child).into_array();

        test_scalar_probe_consistency(&array, &mut array_session().create_execution_ctx());
        Ok(())
    }
}
