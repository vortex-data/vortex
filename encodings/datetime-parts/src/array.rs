// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
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

impl VTable for DateTimeParts {
    type Array = DateTimePartsArray;

    type Metadata = ProstMetadata<DateTimePartsMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &DateTimeParts
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

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

    fn nbuffers(_array: &DateTimePartsArray) -> usize {
        0
    }

    fn buffer(_array: &DateTimePartsArray, idx: usize) -> BufferHandle {
        vortex_panic!("DateTimePartsArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &DateTimePartsArray, idx: usize) -> Option<String> {
        vortex_panic!("DateTimePartsArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(_array: &DateTimePartsArray) -> usize {
        3
    }

    fn child(array: &DateTimePartsArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.days().clone(),
            1 => array.seconds().clone(),
            2 => array.subseconds().clone(),
            _ => vortex_panic!("DateTimePartsArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &DateTimePartsArray, idx: usize) -> String {
        match idx {
            0 => "days".to_string(),
            1 => "seconds".to_string(),
            2 => "subseconds".to_string(),
            _ => vortex_panic!("DateTimePartsArray child_name index {idx} out of bounds"),
        }
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

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            decode_to_temporal(&array, ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
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
}

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

    pub fn into_parts(self) -> DateTimePartsArrayParts {
        DateTimePartsArrayParts {
            dtype: self.dtype,
            days: self.days,
            seconds: self.seconds,
            subseconds: self.subseconds,
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

impl ValidityChild<DateTimeParts> for DateTimeParts {
    fn validity_child(array: &DateTimePartsArray) -> &ArrayRef {
        array.days()
    }
}
