// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical arrays of elapsed-time values.

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Array;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VTable;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::array::TypedArrayRef;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::array::with_empty_buffers;
use crate::array_slots;
use crate::arrays::Dict;
use crate::arrays::Slice;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DurationBuilder;
use crate::dtype::DType;
use crate::dtype::DurationUnit;
use crate::dtype::PType;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar::Scalar;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice, Duration, const { &SliceReduceAdaptor(Duration) });
    kernels.register_reduce_parent_rule(Dict, Duration, const { &TakeReduceAdaptor(Duration) });
}

/// The canonical encoding for duration arrays.
#[derive(Clone, Debug)]
pub struct Duration;

/// An array of elapsed-time values in a shared unit.
///
/// [`DurationArraySlotsExt::values`] returns signed integers in [`DurationArrayExt::unit`].
pub type DurationArray = Array<Duration>;

#[array_slots(Duration)]
pub struct DurationSlots {
    /// Signed integer values in the array's duration unit.
    #[slot(0)]
    pub values: ArrayRef,
}

impl Array<Duration> {
    /// Wraps an `i64` array of duration values without copying the array data.
    pub fn try_new(unit: DurationUnit, values: ArrayRef) -> VortexResult<Self> {
        let len = values.len();
        let dtype = DType::Duration(unit, values.dtype().nullability());
        Self::try_from_parts(ArrayParts::new(
            Duration,
            dtype,
            len,
            EmptyArrayData,
            DurationSlots { values }.into_slots(),
        ))
    }
}

/// Typed access to duration arrays.
pub trait DurationArrayExt: TypedArrayRef<Duration> {
    /// The unit of the stored values.
    fn unit(&self) -> DurationUnit {
        let DType::Duration(unit, _) = self.dtype() else {
            unreachable!("Duration arrays have a duration dtype");
        };
        *unit
    }
}

impl<T: TypedArrayRef<Duration>> DurationArrayExt for T {}

impl VTable for Duration {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.duration");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Duration(_, nullability) = dtype else {
            vortex_bail!("Expected duration dtype, got {dtype}");
        };
        let values = slots
            .get(DurationSlots::VALUES)
            .and_then(Option::as_ref)
            .ok_or_else(|| vortex_err!("Duration values slot is missing"))?;
        vortex_ensure_eq!(slots.len(), 1);
        vortex_ensure_eq!(values.len(), len);
        vortex_ensure_eq!(values.dtype(), &DType::Primitive(PType::I64, *nullability));
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DurationArray buffer index {idx} out of bounds")
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DurationSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<DurationBuilder>() else {
            vortex_bail!("Duration requires a DurationBuilder");
        };
        builder.append_duration(array, ctx)
    }
}

impl ValidityChild<Duration> for Duration {
    fn validity_child(array: ArrayView<'_, Duration>) -> ArrayRef {
        array.values().clone()
    }
}

impl OperationsVTable<Duration> for Duration {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Duration>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let dtype = state.array().dtype().clone();
        let value = state
            .slot(DurationSlots::VALUES)?
            .vortex_expect("Duration values slot is present")
            .execute_scalar(index, ctx)?;
        Scalar::try_new(dtype, value.into_value())
    }

    fn scalar_at(
        array: ArrayView<'_, Duration>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl SliceReduce for Duration {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let values = array.values().slice(range)?;
        Ok(Some(
            DurationArray::try_new(array.unit(), values)?.into_array(),
        ))
    }
}

impl TakeReduce for Duration {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let values = array.values().take(indices.clone())?;
        Ok(Some(
            DurationArray::try_new(array.unit(), values)?.into_array(),
        ))
    }
}

impl ArrayPlugin for Duration {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(array.encoding_id(), ArrayPlugin::id(self));
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let DType::Duration(unit, nullability) = parts.dtype else {
            vortex_bail!("Expected duration dtype, got {}", parts.dtype);
        };
        vortex_ensure_eq!(parts.serialized_id, ArrayPlugin::id(self));
        vortex_ensure_eq!(parts.children.len(), 1);
        vortex_ensure_eq!(parts.metadata.len(), 0);
        let values =
            parts
                .children
                .get(0, &DType::Primitive(PType::I64, *nullability), parts.len)?;
        Ok(DurationArray::try_new(*unit, values)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Duration;
    use super::DurationArray;
    use super::DurationArraySlotsExt;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Dict;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DurationUnit;
    use crate::optimizer::ArrayOptimizer;

    #[test]
    fn take_pushes_down_to_duration_values() -> VortexResult<()> {
        let mut ctx = VortexSession::empty().create_execution_ctx();
        let unit = DurationUnit::Microseconds;
        let values = PrimitiveArray::from_iter([11i64, 0, -7]).into_array();
        let duration = DurationArray::try_new(unit, values)?;
        let indices =
            PrimitiveArray::from_option_iter([Some(2u64), None, Some(0), Some(2)]).into_array();

        let taken = duration.take(indices)?.optimize_ctx(&ctx)?;
        assert!(taken.as_::<Duration>().values().is::<Dict>());

        let expected_values =
            PrimitiveArray::from_option_iter([Some(-7i64), None, Some(11), Some(-7)]).into_array();
        let expected = DurationArray::try_new(unit, expected_values)?;
        assert_arrays_eq!(taken, expected, &mut ctx);
        Ok(())
    }
}
