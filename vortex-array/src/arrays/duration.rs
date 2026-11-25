// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical arrays of signed elapsed-time counts.

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
use crate::arrays::Slice;
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
    session.kernels().register_reduce_parent_rule(
        Slice,
        Duration,
        const { &SliceReduceAdaptor(Duration) },
    );
}

/// The canonical encoding for duration arrays.
#[derive(Clone, Debug)]
pub struct Duration;

/// An array of elapsed-time counts in a shared unit.
pub type DurationArray = Array<Duration>;

#[array_slots(Duration)]
pub struct DurationSlots {
    /// The signed counts.
    #[slot(0)]
    pub counts: ArrayRef,
}

impl Array<Duration> {
    /// Wraps an `i64` array of counts without copying its values.
    pub fn try_new(unit: DurationUnit, counts: ArrayRef) -> VortexResult<Self> {
        let len = counts.len();
        let dtype = DType::Duration(unit, counts.dtype().nullability());
        Self::try_from_parts(ArrayParts::new(
            Duration,
            dtype,
            len,
            EmptyArrayData,
            DurationSlots { counts }.into_slots(),
        ))
    }
}

/// Typed access to duration arrays.
pub trait DurationArrayExt: TypedArrayRef<Duration> {
    /// The unit of the stored counts.
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
        let counts = slots
            .get(DurationSlots::COUNTS)
            .and_then(Option::as_ref)
            .ok_or_else(|| vortex_err!("Duration count array is missing"))?;
        vortex_ensure_eq!(slots.len(), 1);
        vortex_ensure_eq!(counts.len(), len);
        vortex_ensure_eq!(counts.dtype(), &DType::Primitive(PType::I64, *nullability));
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
        array.counts().clone()
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
        let count = state
            .slot(DurationSlots::COUNTS)?
            .vortex_expect("Duration count array is present")
            .execute_scalar(index, ctx)?;
        Scalar::try_new(dtype, count.into_value())
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
        let counts = array.counts().slice(range)?;
        Ok(Some(
            DurationArray::try_new(array.unit(), counts)?.into_array(),
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
        let counts =
            parts
                .children
                .get(0, &DType::Primitive(PType::I64, *nullability), parts.len)?;
        Ok(DurationArray::try_new(*unit, counts)?.into_array())
    }
}
