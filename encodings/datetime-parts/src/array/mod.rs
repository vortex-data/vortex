// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::AnyColumnar;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::TemporalArray;
use vortex_array::block::BlockOp;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::require_child;
use vortex_array::require_spine;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ScaledTemporalParts;
use crate::TemporalParts;
use crate::block::days_block_op;
use crate::canonical::decode_to_temporal;
use crate::split_temporal;
use crate::split_temporal_v2;
use crate::timestamp::temporal_kind;
use crate::timestamp::validate_subseconds_divisor;

mod plugin;

pub use plugin::DateTimePartsPlugin;
pub use plugin::DateTimePartsV2Metadata;
pub use plugin::datetime_parts_v1_id;
pub use plugin::datetime_parts_v2_id;

/// A [`DateTimeParts`]-encoded Vortex array.
pub type DateTimePartsArray = Array<DateTimeParts>;

impl ArrayHash for DateTimePartsData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.subseconds_divisor.hash(state);
    }
}

impl ArrayEq for DateTimePartsData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.subseconds_divisor == other.subseconds_divisor
    }
}

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct DateTimePartsMetadata {
    // Validity lives in the days array
    // TODO(ngates): we should actually model this with a Tuple array when we have one.
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) days_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    pub(crate) seconds_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    pub(crate) subseconds_ptype: i32,
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
    type TypedArrayData = DateTimePartsData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        datetime_parts_v1_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = DateTimePartsSlotsView::from_slots(slots);
        DateTimePartsData::validate(
            dtype,
            slots.days,
            slots.seconds,
            slots.subseconds,
            data.subseconds_divisor,
            len,
        )
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DateTimePartsSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Seconds and subseconds are often constant, so they stay constant to keep the decode
        // fast path.
        let array =
            require_child!(array, array.seconds(), DateTimePartsSlots::SECONDS => AnyColumnar);
        let array = require_child!(
            array,
            array.subseconds(),
            DateTimePartsSlots::SUBSECONDS => AnyColumnar
        );
        // Days last, as the spine: once the time parts are executed, the executor can combine
        // blocks of an encoded days child with them without materializing days.
        let array = require_spine!(array, array.days(), DateTimePartsSlots::DAYS => Primitive);

        let dtype = array.dtype().clone();
        let parts = array.into_parts();

        Ok(ExecutionResult::done(
            decode_to_temporal(parts, &dtype, ctx)?.into_array(),
        ))
    }

    fn block_op(
        array: ArrayView<'_, Self>,
        slot_idx: usize,
        child_dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn BlockOp>>> {
        if slot_idx != DateTimePartsSlots::DAYS {
            return Ok(None);
        }
        Ok(days_block_op(
            array.dtype(),
            child_dtype,
            array.seconds(),
            array.subseconds(),
            array.subseconds_divisor(),
        ))
    }
}

#[array_slots(DateTimeParts)]
pub struct DateTimePartsSlots {
    /// The days component of the datetime, stored as an integer array.
    #[slot(0)]
    pub days: ArrayRef,
    /// The seconds component of the datetime (within the day).
    #[slot(1)]
    pub seconds: ArrayRef,
    /// The sub-second component of the datetime.
    #[slot(2)]
    pub subseconds: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct DateTimePartsData {
    pub(crate) subseconds_divisor: u32,
}

impl DateTimePartsData {
    /// Returns the scaling divisor applied to the `subseconds` child.
    ///
    /// Decoded subseconds are `subseconds * subseconds_divisor`.
    pub fn subseconds_divisor(&self) -> u32 {
        self.subseconds_divisor
    }
}

pub struct DateTimePartsParts {
    pub days: ArrayRef,
    pub seconds: ArrayRef,
    pub subseconds: ArrayRef,
    pub subseconds_divisor: u32,
}

pub trait DateTimePartsOwnedExt {
    fn into_parts(self) -> DateTimePartsParts;
}

impl DateTimePartsOwnedExt for Array<DateTimeParts> {
    fn into_parts(self) -> DateTimePartsParts {
        let subseconds_divisor = self.subseconds_divisor();
        match self.try_into_parts() {
            Ok(parts) => {
                let slots = DateTimePartsSlots::from_slots(parts.slots);
                DateTimePartsParts {
                    days: slots.days,
                    seconds: slots.seconds,
                    subseconds: slots.subseconds,
                    subseconds_divisor,
                }
            }
            Err(array) => {
                let view = array.slots_view();
                DateTimePartsParts {
                    days: view.days.clone(),
                    seconds: view.seconds.clone(),
                    subseconds: view.subseconds.clone(),
                    subseconds_divisor,
                }
            }
        }
    }
}

impl Display for DateTimePartsData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.subseconds_divisor != 1 {
            write!(f, "subseconds_divisor: {}", self.subseconds_divisor)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DateTimeParts;

impl DateTimeParts {
    /// Construct a new [`DateTimePartsArray`] from its components with `subseconds_divisor = 1`.
    pub fn try_new(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
    ) -> VortexResult<DateTimePartsArray> {
        Self::try_new_with_divisor(dtype, days, seconds, subseconds, 1)
    }

    /// Construct a new [`DateTimePartsArray`] from its components and subseconds divisor.
    pub fn try_new_with_divisor(
        dtype: DType,
        days: ArrayRef,
        seconds: ArrayRef,
        subseconds: ArrayRef,
        subseconds_divisor: u32,
    ) -> VortexResult<DateTimePartsArray> {
        let len = days.len();
        DateTimePartsData::validate(
            &dtype,
            &days,
            &seconds,
            &subseconds,
            subseconds_divisor,
            len,
        )?;
        let slots = smallvec![Some(days), Some(seconds), Some(subseconds)];
        let data = DateTimePartsData { subseconds_divisor };
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(DateTimeParts, dtype, len, data, slots))
        })
    }

    /// Construct a [`DateTimePartsArray`] from a [`TemporalArray`] with unscaled subseconds
    /// (`subseconds_divisor = 1`).
    pub fn try_from_temporal(
        temporal: TemporalArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DateTimePartsArray> {
        let dtype = temporal.dtype().clone();
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal, ctx)?;
        Self::try_new(dtype, days, seconds, subseconds)
    }

    /// Construct a [`DateTimePartsArray`] from a [`TemporalArray`], scaling coarse-grained
    /// subseconds by a power-of-1000 divisor when lossless.
    pub fn try_from_temporal_v2(
        temporal: TemporalArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<DateTimePartsArray> {
        let dtype = temporal.dtype().clone();
        let ScaledTemporalParts {
            days,
            seconds,
            subseconds,
            subseconds_divisor,
        } = split_temporal_v2(temporal, ctx)?;
        Self::try_new_with_divisor(dtype, days, seconds, subseconds, subseconds_divisor)
    }
}

impl DateTimePartsData {
    pub fn validate(
        dtype: &DType,
        days: &ArrayRef,
        seconds: &ArrayRef,
        subseconds: &ArrayRef,
        subseconds_divisor: u32,
        len: usize,
    ) -> VortexResult<()> {
        let kind = temporal_kind(dtype)?;
        validate_subseconds_divisor(kind.unit(), subseconds_divisor)?;

        vortex_ensure_eq!(days.len(), len);

        if !days.dtype().is_int() || (dtype.is_nullable() != days.dtype().is_nullable()) {
            vortex_bail!(
                "Expected integer with nullability {}, got {}",
                dtype.is_nullable(),
                days.dtype()
            );
        }
        if !seconds.dtype().is_int() || seconds.dtype().is_nullable() {
            vortex_bail!(
                "expected type: non-nullable integer but instead got {}",
                seconds.dtype()
            );
        }
        if !subseconds.dtype().is_int() || subseconds.dtype().is_nullable() {
            vortex_bail!(
                "expected type: non-nullable integer but instead got {}",
                subseconds.dtype()
            );
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
