// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

use vortex_array::Array;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::VisitorVTable;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::compute::rules::PARENT_RULES;

vtable!(DateTimeParts);

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

impl VTable for DateTimePartsVTable {
    type Array = DateTimePartsArray;

    type Metadata = ProstMetadata<DateTimePartsMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.datetimeparts")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DateTimePartsVTable.as_vtable()
    }

    fn metadata(array: &DateTimePartsArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DateTimePartsMetadata {
            days_ptype: PType::try_from(array.days().dtype())? as i32,
            seconds_ptype: PType::try_from(array.seconds().dtype())? as i32,
            subseconds_ptype: PType::try_from(array.subseconds().dtype())? as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<DateTimePartsMetadata> as DeserializeMetadata>::deserialize(buffer)?,
        ))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DateTimePartsArray> {
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

        DateTimePartsArray::try_new(dtype.clone(), days, seconds, subseconds)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 3,
            "DateTimePartsArray expects exactly 3 children (days, seconds, subseconds), got {}",
            children.len()
        );

        let mut children_iter = children.into_iter();
        array.days = children_iter.next().vortex_expect("checked");
        array.seconds = children_iter.next().vortex_expect("checked");
        array.subseconds = children_iter.next().vortex_expect("checked");

        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing all components preserves values
        Ok(Some(unsafe {
            DateTimePartsArray::new_unchecked(
                array.dtype().clone(),
                array.days().slice(range.clone()),
                array.seconds().slice(range.clone()),
                array.subseconds().slice(range),
            )
            .into_array()
        }))
    }
}

#[derive(Clone, Debug)]
pub struct DateTimePartsArray {
    dtype: DType,
    days: ArrayRef,
    seconds: ArrayRef,
    subseconds: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct DateTimePartsVTable;

impl DateTimePartsArray {
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
            days,
            seconds,
            subseconds,
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
            days,
            seconds,
            subseconds,
            stats_set: Default::default(),
        }
    }

    pub fn days(&self) -> &ArrayRef {
        &self.days
    }

    pub fn seconds(&self) -> &ArrayRef {
        &self.seconds
    }

    pub fn subseconds(&self) -> &ArrayRef {
        &self.subseconds
    }
}

impl BaseArrayVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn len(array: &DateTimePartsArray) -> usize {
        array.days.len()
    }

    fn dtype(array: &DateTimePartsArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DateTimePartsArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &DateTimePartsArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.days.array_hash(state, precision);
        array.seconds.array_hash(state, precision);
        array.subseconds.array_hash(state, precision);
    }

    fn array_eq(
        array: &DateTimePartsArray,
        other: &DateTimePartsArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array.days.array_eq(&other.days, precision)
            && array.seconds.array_eq(&other.seconds, precision)
            && array.subseconds.array_eq(&other.subseconds, precision)
    }
}

impl ValidityChild<DateTimePartsVTable> for DateTimePartsVTable {
    fn validity_child(array: &DateTimePartsArray) -> &ArrayRef {
        array.days()
    }
}

impl VisitorVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn visit_buffers(_array: &DateTimePartsArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DateTimePartsArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("days", array.days());
        visitor.visit_child("seconds", array.seconds());
        visitor.visit_child("subseconds", array.subseconds());
    }
}
