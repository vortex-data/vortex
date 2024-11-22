//! Array validity and nullability behavior, used by arrays and compute functions.

use std::fmt::{Debug, Display};
use std::ops::BitAnd;

use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, Nullability};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexError, VortexExpect as _, VortexResult,
};

use crate::array::{BoolArray, ConstantArray};
use crate::compute::unary::scalar_at;
use crate::compute::{filter, slice, take, FilterMask, TakeOptions};
use crate::encoding::Encoding;
use crate::stats::ArrayStatistics;
use crate::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};

pub trait ValidityVTable<Array> {
    // TODO(ngates): can we implement this based on logical validity? Or is that too expensive?
    fn is_valid(&self, array: &Array, index: usize) -> bool;
    fn logical_validity(&self, array: &Array) -> LogicalValidity;
}

impl<E: Encoding> ValidityVTable<ArrayData> for E
where
    E: ValidityVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn is_valid(&self, array: &ArrayData, index: usize) -> bool {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::is_valid(encoding, array_ref, index)
    }

    fn logical_validity(&self, array: &ArrayData) -> LogicalValidity {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        ValidityVTable::logical_validity(encoding, array_ref)
    }
}

pub trait ArrayValidity {
    fn is_valid(&self, index: usize) -> bool;
    fn logical_validity(&self) -> LogicalValidity;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ValidityMetadata {
    NonNullable,
    AllValid,
    AllInvalid,
    Array,
}

impl Display for ValidityMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ValidityMetadata {
    pub fn to_validity<F>(&self, array_fn: F) -> Validity
    where
        F: FnOnce() -> ArrayData,
    {
        match self {
            Self::NonNullable => Validity::NonNullable,
            Self::AllValid => Validity::AllValid,
            Self::AllInvalid => Validity::AllInvalid,
            Self::Array => Validity::Array(array_fn()),
        }
    }
}

/// Validity information for an array
#[derive(Clone, Debug)]
pub enum Validity {
    /// Items *can't* be null
    NonNullable,
    /// All items are valid
    AllValid,
    /// All items are null
    AllInvalid,
    /// Specified items are null
    Array(ArrayData),
}

impl Validity {
    /// The [`DType`] of the underlying validity array (if it exists).
    pub const DTYPE: DType = DType::Bool(Nullability::NonNullable);

    pub fn to_metadata(&self, length: usize) -> VortexResult<ValidityMetadata> {
        match self {
            Self::NonNullable => Ok(ValidityMetadata::NonNullable),
            Self::AllValid => Ok(ValidityMetadata::AllValid),
            Self::AllInvalid => Ok(ValidityMetadata::AllInvalid),
            Self::Array(a) => {
                // We force the caller to validate the length here.
                let validity_len = a.len();
                if validity_len != length {
                    vortex_bail!(
                        "Validity array length {} doesn't match array length {}",
                        validity_len,
                        length
                    )
                }
                Ok(ValidityMetadata::Array)
            }
        }
    }

    pub fn null_count(&self, length: usize) -> VortexResult<usize> {
        match self {
            Self::NonNullable | Self::AllValid => Ok(0),
            Self::AllInvalid => Ok(length),
            Self::Array(a) => {
                let validity_len = a.len();
                if validity_len != length {
                    vortex_bail!(
                        "Validity array length {} doesn't match array length {}",
                        validity_len,
                        length
                    )
                }
                let true_count = a.statistics().compute_true_count().ok_or_else(|| {
                    vortex_err!("Failed to compute true count from validity array")
                })?;
                Ok(length - true_count)
            }
        }
    }

    /// If Validity is [`Validity::Array`], returns the array, otherwise returns `None`.
    pub fn into_array(self) -> Option<ArrayData> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// If Validity is [`Validity::Array`], returns a reference to the array array, otherwise returns `None`.
    pub fn as_array(&self) -> Option<&ArrayData> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn nullability(&self) -> Nullability {
        match self {
            Self::NonNullable => Nullability::NonNullable,
            _ => Nullability::Nullable,
        }
    }

    /// Returns whether the `index` item is valid.
    #[inline]
    pub fn is_valid(&self, index: usize) -> bool {
        match self {
            Self::NonNullable | Self::AllValid => true,
            Self::AllInvalid => false,
            Self::Array(a) => scalar_at(a, index)
                .and_then(|s| bool::try_from(&s))
                .unwrap_or_else(|err| {
                    vortex_panic!(
                        err,
                        "Failed to get bool from Validity Array at index {}",
                        index
                    )
                }),
        }
    }

    #[inline]
    pub fn is_null(&self, index: usize) -> bool {
        !self.is_valid(index)
    }

    pub fn slice(&self, start: usize, stop: usize) -> VortexResult<Self> {
        match self {
            Self::Array(a) => Ok(Self::Array(slice(a, start, stop)?)),
            _ => Ok(self.clone()),
        }
    }

    pub fn take(&self, indices: &ArrayData, options: TakeOptions) -> VortexResult<Self> {
        match self {
            Self::NonNullable => Ok(Self::NonNullable),
            Self::AllValid => Ok(Self::AllValid),
            Self::AllInvalid => Ok(Self::AllInvalid),
            Self::Array(a) => Ok(Self::Array(take(a, indices, options)?)),
        }
    }

    pub fn filter(&self, mask: &FilterMask) -> VortexResult<Self> {
        // NOTE(ngates): we take the mask as a reference to avoid the caller cloning unnecessarily
        //  if we happen to be NonNullable, AllValid, or AllInvalid.
        match self {
            v @ (Validity::NonNullable | Validity::AllValid | Validity::AllInvalid) => {
                Ok(v.clone())
            }
            Validity::Array(arr) => Ok(Validity::Array(filter(arr, mask.clone())?)),
        }
    }

    pub fn to_logical(&self, length: usize) -> LogicalValidity {
        match self {
            Self::NonNullable => LogicalValidity::AllValid(length),
            Self::AllValid => LogicalValidity::AllValid(length),
            Self::AllInvalid => LogicalValidity::AllInvalid(length),
            Self::Array(a) => {
                // Logical validity should map into AllValid/AllInvalid where possible.
                if a.statistics().compute_min::<bool>().unwrap_or(false) {
                    LogicalValidity::AllValid(length)
                } else if a
                    .statistics()
                    .compute_max::<bool>()
                    .map(|m| !m)
                    .unwrap_or(false)
                {
                    LogicalValidity::AllInvalid(length)
                } else {
                    LogicalValidity::Array(a.clone())
                }
            }
        }
    }

    /// Logically & two Validity values of the same length
    pub fn and(self, rhs: Validity) -> VortexResult<Validity> {
        let validity = match (self, rhs) {
            // Should be pretty clear
            (Validity::NonNullable, Validity::NonNullable) => Validity::NonNullable,
            // Any `AllInvalid` makes the output all invalid values
            (Validity::AllInvalid, _) | (_, Validity::AllInvalid) => Validity::AllInvalid,
            // All truthy values on one side, which makes no effect on an `Array` variant
            (Validity::Array(a), Validity::AllValid)
            | (Validity::Array(a), Validity::NonNullable)
            | (Validity::NonNullable, Validity::Array(a))
            | (Validity::AllValid, Validity::Array(a)) => Validity::Array(a),
            // Both sides are all valid
            (Validity::NonNullable, Validity::AllValid)
            | (Validity::AllValid, Validity::NonNullable)
            | (Validity::AllValid, Validity::AllValid) => Validity::AllValid,
            // Here we actually have to do some work
            (Validity::Array(lhs), Validity::Array(rhs)) => {
                let lhs = BoolArray::try_from(lhs)?;
                let rhs = BoolArray::try_from(rhs)?;

                let lhs = lhs.boolean_buffer();
                let rhs = rhs.boolean_buffer();

                Validity::from(lhs.bitand(&rhs))
            }
        };

        Ok(validity)
    }

    pub fn patch<P: AsPrimitive<usize>>(
        self,
        len: usize,
        positions: &[P],
        patches: Validity,
    ) -> VortexResult<Self> {
        if let Some(last_pos) = positions.last() {
            if last_pos.as_() >= len {
                vortex_bail!(OutOfBounds: last_pos.as_(), 0, len)
            }
        }

        if matches!(self, Validity::NonNullable | Validity::AllValid)
            && matches!(patches, Validity::NonNullable | Validity::AllValid)
            || self == patches
        {
            return Ok(self);
        }

        let source = match self {
            Validity::NonNullable => BoolArray::from(BooleanBuffer::new_set(len)),
            Validity::AllValid => BoolArray::from(BooleanBuffer::new_set(len)),
            Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(len)),
            Validity::Array(a) => a.into_bool()?,
        };

        let patch_values = match patches {
            Validity::NonNullable => BoolArray::from(BooleanBuffer::new_set(positions.len())),
            Validity::AllValid => BoolArray::from(BooleanBuffer::new_set(positions.len())),
            Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(positions.len())),
            Validity::Array(a) => a.into_bool()?,
        };

        Validity::try_from(source.patch(positions, patch_values)?.into_array())
    }

    /// Convert into a nullable variant
    pub fn into_nullable(self) -> Validity {
        match self {
            Self::NonNullable => Self::AllValid,
            _ => self,
        }
    }
}

impl PartialEq for Validity {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NonNullable, Self::NonNullable) => true,
            (Self::AllValid, Self::AllValid) => true,
            (Self::AllInvalid, Self::AllInvalid) => true,
            (Self::Array(a), Self::Array(b)) => {
                let a_buffer = a
                    .clone()
                    .into_bool()
                    .vortex_expect("Failed to get Validity Array as BoolArray")
                    .boolean_buffer();
                let b_buffer = b
                    .clone()
                    .into_bool()
                    .vortex_expect("Failed to get Validity Array as BoolArray")
                    .boolean_buffer();
                a_buffer == b_buffer
            }
            _ => false,
        }
    }
}

impl From<BooleanBuffer> for Validity {
    fn from(value: BooleanBuffer) -> Self {
        if value.count_set_bits() == value.len() {
            Self::AllValid
        } else if value.count_set_bits() == 0 {
            Self::AllInvalid
        } else {
            Self::Array(BoolArray::from(value).into_array())
        }
    }
}

impl From<NullBuffer> for Validity {
    fn from(value: NullBuffer) -> Self {
        value.into_inner().into()
    }
}

impl TryFrom<ArrayData> for Validity {
    type Error = VortexError;

    fn try_from(value: ArrayData) -> Result<Self, Self::Error> {
        LogicalValidity::try_from(value).map(|a| a.into_validity())
    }
}

impl FromIterator<LogicalValidity> for Validity {
    fn from_iter<T: IntoIterator<Item = LogicalValidity>>(iter: T) -> Self {
        let validities: Vec<LogicalValidity> = iter.into_iter().collect();

        // If they're all valid, then return a single validity.
        if validities.iter().all(|v| v.all_valid()) {
            return Self::AllValid;
        }
        // If they're all invalid, then return a single invalidity.
        if validities.iter().all(|v| v.all_invalid()) {
            return Self::AllInvalid;
        }

        // Else, construct the boolean buffer
        let mut buffer = BooleanBufferBuilder::new(validities.iter().map(|v| v.len()).sum());
        for validity in validities {
            match validity {
                LogicalValidity::AllValid(count) => buffer.append_n(count, true),
                LogicalValidity::AllInvalid(count) => buffer.append_n(count, false),
                LogicalValidity::Array(array) => {
                    let array_buffer = array
                        .into_bool()
                        .vortex_expect("Failed to get Validity Array as BoolArray")
                        .boolean_buffer();
                    buffer.append_buffer(&array_buffer);
                }
            };
        }
        let bool_array = BoolArray::from(buffer.finish());
        Self::Array(bool_array.into_array())
    }
}

impl FromIterator<bool> for Validity {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Self::Array(BoolArray::from_iter(iter).into_array())
    }
}

#[derive(Clone, Debug)]
pub enum LogicalValidity {
    AllValid(usize),
    AllInvalid(usize),
    Array(ArrayData),
}

impl LogicalValidity {
    pub fn try_new_from_array(array: ArrayData) -> VortexResult<Self> {
        if !matches!(array.dtype(), &Validity::DTYPE) {
            vortex_bail!("Expected a non-nullable boolean array");
        }

        let true_count = array.statistics().compute_true_count().ok_or_else(|| {
            vortex_err!(
                "Failed to compute true count from validity array {:#?}",
                array
            )
        })?;
        if true_count == array.len() {
            return Ok(Self::AllValid(array.len()));
        } else if true_count == 0 {
            return Ok(Self::AllInvalid(array.len()));
        }

        Ok(Self::Array(array))
    }

    pub fn to_null_buffer(&self) -> VortexResult<Option<NullBuffer>> {
        match self {
            Self::AllValid(_) => Ok(None),
            Self::AllInvalid(l) => Ok(Some(NullBuffer::new_null(*l))),
            Self::Array(a) => Ok(Some(NullBuffer::new(
                a.clone().into_bool()?.boolean_buffer(),
            ))),
        }
    }

    pub fn all_valid(&self) -> bool {
        matches!(self, Self::AllValid(_))
    }

    pub fn all_invalid(&self) -> bool {
        matches!(self, Self::AllInvalid(_))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::AllValid(n) => *n,
            Self::AllInvalid(n) => *n,
            Self::Array(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::AllValid(n) => *n == 0,
            Self::AllInvalid(n) => *n == 0,
            Self::Array(a) => a.is_empty(),
        }
    }

    pub fn into_validity(self) -> Validity {
        match self {
            Self::AllValid(_) => Validity::AllValid,
            Self::AllInvalid(_) => Validity::AllInvalid,
            Self::Array(a) => Validity::Array(a),
        }
    }

    pub fn null_count(&self, length: usize) -> VortexResult<usize> {
        match self {
            Self::AllValid(_) => Ok(0),
            Self::AllInvalid(_) => Ok(length),
            Self::Array(a) => {
                let validity_len = a.len();
                if validity_len != length {
                    vortex_bail!(
                        "Validity array length {} doesn't match array length {}",
                        validity_len,
                        length
                    )
                }
                let true_count = a.statistics().compute_true_count().ok_or_else(|| {
                    vortex_err!("Failed to compute true count from validity array")
                })?;
                Ok(length - true_count)
            }
        }
    }
}

impl TryFrom<ArrayData> for LogicalValidity {
    type Error = VortexError;

    fn try_from(array: ArrayData) -> VortexResult<Self> {
        Self::try_new_from_array(array)
    }
}

impl IntoArrayData for LogicalValidity {
    fn into_array(self) -> ArrayData {
        match self {
            Self::AllValid(len) => ConstantArray::new(true, len).into_array(),
            Self::AllInvalid(len) => ConstantArray::new(false, len).into_array(),
            Self::Array(a) => a,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::array::BoolArray;
    use crate::validity::Validity;
    use crate::IntoArrayData;

    #[rstest]
    #[case(Validity::NonNullable, 5, &[2, 4], Validity::NonNullable, Validity::NonNullable)]
    #[case(Validity::NonNullable, 5, &[2, 4], Validity::AllValid, Validity::NonNullable)]
    #[case(Validity::NonNullable, 5, &[2, 4], Validity::AllInvalid, Validity::Array(BoolArray::from_iter([true, true, false, true, false]).into_array())
    )]
    #[case(Validity::NonNullable, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([true, true, true, true, false]).into_array())
    )]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::NonNullable, Validity::AllValid)]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllValid, Validity::AllValid)]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllInvalid, Validity::Array(BoolArray::from_iter([true, true, false, true, false]).into_array())
    )]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([true, true, true, true, false]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::NonNullable, Validity::Array(BoolArray::from_iter([false, false, true, false, true]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllValid, Validity::Array(BoolArray::from_iter([false, false, true, false, true]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllInvalid, Validity::AllInvalid)]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([false, false, true, false, false]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::NonNullable, Validity::Array(BoolArray::from_iter([false, true, true, true, true]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::AllValid, Validity::Array(BoolArray::from_iter([false, true, true, true, true]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::AllInvalid, Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array())
    )]
    #[case(Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()), 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([false, true, true, true, false]).into_array())
    )]
    fn patch_validity(
        #[case] validity: Validity,
        #[case] len: usize,
        #[case] positions: &[usize],
        #[case] patches: Validity,
        #[case] expected: Validity,
    ) {
        assert_eq!(validity.patch(len, positions, patches).unwrap(), expected);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_patch() {
        Validity::NonNullable
            .patch(2, &[4], Validity::AllInvalid)
            .unwrap();
    }
}
