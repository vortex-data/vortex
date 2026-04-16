// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

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
use vortex_array::TypedArrayRef;
use vortex_array::arrays::TemporalArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
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
use vortex_session::registry::CachedId;

use crate::TemporalParts;
use crate::canonical::decode_to_temporal;
use crate::compute::kernel::PARENT_KERNELS;
use crate::compute::rules::PARENT_RULES;
use crate::split_temporal;

/// A [`DateTimeParts`]-encoded Vortex array.
pub type DateTimePartsArray = Array<DateTimeParts>;

impl ArrayHash for DateTimePartsData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for DateTimePartsData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

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
        static ID: CachedId = CachedId::new("vortex.datetimeparts");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let days = slots[DAYS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray days slot");
        let seconds = slots[SECONDS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray seconds slot");
        let subseconds = slots[SUBSECONDS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray subseconds slot");
        DateTimePartsData::validate(dtype, days, seconds, subseconds, len)
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
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
    ) -> VortexResult<ArrayParts<Self>> {
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

        let slots = vec![Some(days), Some(seconds), Some(subseconds)];
        let data = DateTimePartsData {};
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
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
pub struct DateTimePartsData {}

impl Display for DateTimePartsData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait DateTimePartsArrayExt: TypedArrayRef<DateTimeParts> {
    fn days(&self) -> &ArrayRef {
        self.as_ref().slots()[DAYS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray days slot")
    }

    fn seconds(&self) -> &ArrayRef {
        self.as_ref().slots()[SECONDS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray seconds slot")
    }

    fn subseconds(&self) -> &ArrayRef {
        self.as_ref().slots()[SUBSECONDS_SLOT]
            .as_ref()
            .vortex_expect("DateTimePartsArray subseconds slot")
    }
}

impl<T: TypedArrayRef<DateTimeParts>> DateTimePartsArrayExt for T {}

#[derive(Clone, Debug)]
pub struct DateTimeParts;

impl DateTimeParts {
    /// Construct a new [`DateTimePartsArray`] from its components.
    pub fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<DateTimePartsArray> {
        let len = days.len();
        DateTimePartsData::validate(&dtype, &days, &seconds, &subseconds, len)?;
        let slots = vec![Some(days), Some(seconds), Some(subseconds)];
        let data = DateTimePartsData {};
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(DateTimeParts, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Construct a [`DateTimePartsArray`] from a [`TemporalArray`].
    pub fn try_from_temporal(temporal: TemporalArray) -> VortexResult<DateTimePartsArray> {
        let dtype = temporal.dtype().clone();
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal)?;
        Self::try_new(dtype, days, seconds, subseconds)
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
}

impl ValidityChild<DateTimeParts> for DateTimeParts {
    fn validity_child(array: ArrayView<'_, DateTimeParts>) -> ArrayRef {
        array.days().clone()
    }
}
