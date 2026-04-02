// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::TemporalArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
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

    type Metadata = ProstMetadata<DateTimePartsMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &DateTimeParts
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &DateTimePartsData) -> usize {
        array.days().len()
    }

    fn dtype(array: &DateTimePartsData) -> &DType {
        &array.dtype
    }

    fn stats(array: &DateTimePartsData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &DateTimePartsData,
        state: &mut H,
        precision: Precision,
    ) {
        array.days().array_hash(state, precision);
        array.seconds().array_hash(state, precision);
        array.subseconds().array_hash(state, precision);
    }

    fn array_eq(
        array: &DateTimePartsData,
        other: &DateTimePartsData,
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

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DateTimePartsMetadata {
            days_ptype: PType::try_from(array.days().dtype())? as i32,
            seconds_ptype: PType::try_from(array.seconds().dtype())? as i32,
            subseconds_ptype: PType::try_from(array.subseconds().dtype())? as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<DateTimePartsMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DateTimePartsData> {
        if children.len() != 3 {
            vortex_bail!(
                "Expected 3 children for datetime-parts encoding, found {}",
                children.len()
            )
        }

        let days = children.get(
            0,
            &DType::Primitive(metadata.0.get_days_ptype()?, dtype.nullability()),
            len,
        )?;
        let seconds = children.get(
            1,
            &DType::Primitive(metadata.0.get_seconds_ptype()?, Nullability::NonNullable),
            len,
        )?;
        let subseconds = children.get(
            2,
            &DType::Primitive(metadata.0.get_subseconds_ptype()?, Nullability::NonNullable),
            len,
        )?;

        DateTimePartsData::try_new(dtype.clone(), days, seconds, subseconds)
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "DateTimePartsArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
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
    dtype: DType,
    pub(super) slots: Vec<Option<ArrayRef>>,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct DateTimePartsArrayParts {
    pub dtype: DType,
    pub days: ArrayRef,
    pub seconds: ArrayRef,
    pub subseconds: ArrayRef,
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
        Array::try_from_data(DateTimePartsData::try_new(
            dtype, days, seconds, subseconds,
        )?)
    }

    /// Construct a [`DateTimePartsArray`] from a [`TemporalArray`].
    pub fn try_from_temporal(temporal: TemporalArray) -> VortexResult<DateTimePartsArray> {
        Array::try_from_data(DateTimePartsData::try_from(temporal)?)
    }
}

impl DateTimePartsData {
    pub fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<Self> {
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

        let length = days.len();
        if length != seconds.len() || length != subseconds.len() {
            vortex_bail!(
                "Mismatched lengths {} {} {}",
                days.len(),
                seconds.len(),
                subseconds.len()
            );
        }

        Ok(Self {
            dtype,
            slots: vec![Some(days), Some(seconds), Some(subseconds)],
            stats_set: Default::default(),
        })
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> Self {
        Self {
            dtype,
            slots: vec![Some(days), Some(seconds), Some(subseconds)],
            stats_set: Default::default(),
        }
    }

    pub fn into_parts(mut self) -> DateTimePartsArrayParts {
        DateTimePartsArrayParts {
            dtype: self.dtype,
            days: self.slots[DAYS_SLOT]
                .take()
                .vortex_expect("DateTimePartsArray days slot"),
            seconds: self.slots[SECONDS_SLOT]
                .take()
                .vortex_expect("DateTimePartsArray seconds slot"),
            subseconds: self.slots[SUBSECONDS_SLOT]
                .take()
                .vortex_expect("DateTimePartsArray subseconds slot"),
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.days().len()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.days().len() == 0
    }

    /// Returns the logical data type of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
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
