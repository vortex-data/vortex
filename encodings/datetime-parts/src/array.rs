// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::TemporalArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::canonical::decode_to_temporal;
use crate::compute::kernel::PARENT_KERNELS;
use crate::compute::rules::PARENT_RULES;

vtable!(DateTimeParts, DateTimeParts, DateTimePartsData);

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    #[prost(enumeration = "PType", tag = "1")]
    pub days_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    pub seconds_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    pub subseconds_ptype: i32,
}

impl DateTimePartsMetadata {
    pub fn get_days_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.days_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.days_ptype))
    }

    pub fn get_seconds_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.seconds_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.seconds_ptype))
    }

    pub fn get_subseconds_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.subseconds_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.subseconds_ptype))
    }
}

impl VTable for DateTimeParts {
    type ArrayData = DateTimePartsData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        DateTimePartsData::validate(dtype, data.days(), data.seconds(), data.subseconds(), len)
    }

    fn array_hash<H: std::hash::Hasher>(
        array: ArrayView<'_, Self>,
        state: &mut H,
        precision: Precision,
    ) {
        array.days().array_hash(state, precision);
        array.seconds().array_hash(state, precision);
        array.subseconds().array_hash(state, precision);
    }

    fn array_eq(
        array: ArrayView<'_, Self>,
        other: ArrayView<'_, Self>,
        precision: Precision,
    ) -> bool {
        array.days().array_eq(other.days(), precision)
            && array.seconds().array_eq(other.seconds(), precision)
            && array.subseconds().array_eq(other.subseconds(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DateTimePartsArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("DateTimePartsArray buffer_name index {idx} out of bounds")
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DateTimePartsMetadata {
                days_ptype: PType::try_from(array.days().dtype())? as i32,
                seconds_ptype: PType::try_from(array.seconds().dtype())? as i32,
                subseconds_ptype: PType::try_from(array.subseconds().dtype())? as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<DateTimePartsData> {
        let metadata = DateTimePartsMetadata::decode(metadata)?;
        if children.len() != 3 {
            vortex_bail!(
                "Expected 3 children for datetime-parts encoding, found {}",
                children.len()
            )
        }

        let days = children.get(
            0,
            &DType::Primitive(metadata.get_days_ptype()?, dtype.nullability()),
            len,
        )?;
        let seconds = children.get(
            1,
            &DType::Primitive(metadata.get_seconds_ptype()?, Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            2,
            &DType::Primitive(metadata.get_subseconds_ptype()?, Nullability::NonNullable),
            len,
        )?;

        DateTimePartsData::try_new(dtype.clone(), days, seconds, subseconds)
    }

    fn infer_slots(data: &Self::ArrayData) -> Vec<Option<ArrayRef>> {
        data.slots.clone()
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        array.slots()
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decode_to_temporal(&array, ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

/// The days component of the datetime, stored as an integer array.
pub(super) const DAYS_SLOT: usize = 0;
/// The seconds component of the datetime (within the day).
pub(super) const SECONDS_SLOT: usize = 1;
/// The sub-second component of the datetime.
pub(super) const SUBSECONDS_SLOT: usize = 2;
pub(super) const NUM_SLOTS: usize = 3;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["days", "seconds", "subseconds"];

#[derive(Clone, Debug)]
pub struct DateTimePartsData {
    pub(super) slots: Vec<Option<ArrayRef>>,
}

#[derive(Clone, Debug)]
pub struct DateTimeParts;

impl DateTimeParts {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.datetimeparts");

    /// Construct a new [`DateTimePartsArray`] from its components.
    pub fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<DateTimePartsArray> {
        let data = DateTimePartsData::try_new(dtype.clone(), days, seconds, subseconds)?;
        let len = data.len();
        Ok(
            unsafe {
                Array::from_parts_unchecked(ArrayParts::new(DateTimeParts, dtype, len, data))
            },
        )
    }

    /// Construct a [`DateTimePartsArray`] from a [`TemporalArray`].
    pub fn try_from_temporal(temporal: TemporalArray) -> VortexResult<DateTimePartsArray> {
        let dtype = temporal.dtype().clone();
        let data = DateTimePartsData::try_from(temporal)?;
        let len = data.len();
        Ok(
            unsafe {
                Array::from_parts_unchecked(ArrayParts::new(DateTimeParts, dtype, len, data))
            },
        )
    }
}

impl DateTimePartsData {
    pub fn validate(
        dtype: &DType,
        days: &ArrayRef,
        seconds: &ArrayRef,
        subseconds: &ArrayRef,
        len: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(days.len() == len, "expected len {len}, got {}", days.len());

        if !days.dtype().is_int() || (dtype.is_nullable() != days.dtype().is_nullable()) {
            vortex_bail!(
                "Expected integer with nullability {}, got {}",
                dtype.is_nullable(),
                days.dtype()
            );
        }
        if !seconds.dtype().is_int() || seconds.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable integer", seconds.dtype());
        }
        if !subseconds.dtype().is_int() || subseconds.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable integer", subseconds.dtype());
        }

        if len != seconds.len() || len != subseconds.len() {
            vortex_bail!(
                "Mismatched lengths {} {} {}",
                days.len(),
                seconds.len(),
                subseconds.len()
            );
        }

        Ok(())
    }

    pub(crate) fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<Self> {
        Self::validate(&dtype, &days, &seconds, &subseconds, days.len())?;
        Ok(Self {
            slots: vec![Some(days), Some(seconds), Some(subseconds)],
        })
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.days().len()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.days().len() == 0
    }

    pub fn days(&self) -> &ArrayRef {
        self.slots[DAYS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray days slot")
    }

    pub fn seconds(&self) -> &ArrayRef {
        self.slots[SECONDS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray seconds slot")
    }

    pub fn subseconds(&self) -> &ArrayRef {
        self.slots[SUBSECONDS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray subseconds slot")
    }
}

impl ValidityChild<DateTimeParts> for DateTimeParts {
    fn validity_child(array: &DateTimePartsData) -> &ArrayRef {
        array.days()
    }
}
