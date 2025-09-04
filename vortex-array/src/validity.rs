// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array validity and nullability behavior, used by arrays and compute functions.

use std::fmt::Debug;
use std::ops::{BitAnd, Not, Range};

use arrow_buffer::{BooleanBuffer, NullBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect as _, VortexResult, vortex_err, vortex_panic};
use vortex_mask::{AllOr, Mask, MaskValues};
use vortex_scalar::Scalar;

use crate::arrays::{BoolArray, ConstantArray};
use crate::compute::{fill_null, filter, sum, take};
use crate::patches::Patches;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

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
    Array(ArrayRef),
}

impl Validity {
    /// The [`DType`] of the underlying validity array (if it exists).
    pub const DTYPE: DType = DType::Bool(Nullability::NonNullable);

    /// If Validity is [`Validity::Array`], returns the array, otherwise returns `None`.
    #[inline]
    pub fn into_array(self) -> Option<ArrayRef> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// If Validity is [`Validity::Array`], returns a reference to the array array, otherwise returns `None`.
    #[inline]
    pub fn as_array(&self) -> Option<&ArrayRef> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }

    #[inline]
    pub fn nullability(&self) -> Nullability {
        if matches!(self, Self::NonNullable) {
            Nullability::NonNullable
        } else {
            Nullability::Nullable
        }
    }

    /// The union nullability and validity.
    #[inline]
    pub fn union_nullability(self, nullability: Nullability) -> Self {
        match nullability {
            Nullability::NonNullable => self,
            Nullability::Nullable => self.into_nullable(),
        }
    }

    #[inline]
    pub fn all_valid(&self) -> bool {
        match self {
            Validity::NonNullable | Validity::AllValid => true,
            Validity::AllInvalid => false,
            Validity::Array(array) => {
                usize::try_from(&sum(array).vortex_expect("must have sum for bool array"))
                    .vortex_expect("sum must be a usize")
                    == array.len()
            }
        }
    }

    #[inline]
    pub fn all_invalid(&self) -> bool {
        match self {
            Validity::NonNullable | Validity::AllValid => false,
            Validity::AllInvalid => true,
            Validity::Array(array) => {
                usize::try_from(&sum(array).vortex_expect("must have sum for bool array"))
                    .vortex_expect("sum must be a usize")
                    == 0
            }
        }
    }

    /// Returns whether the `index` item is valid.
    #[inline]
    pub fn is_valid(&self, index: usize) -> bool {
        match self {
            Self::NonNullable | Self::AllValid => true,
            Self::AllInvalid => false,
            Self::Array(a) => {
                let scalar = a.scalar_at(index);
                scalar
                    .as_bool()
                    .value()
                    .vortex_expect("Validity must be non-nullable")
            }
        }
    }

    #[inline]
    pub fn is_null(&self, index: usize) -> bool {
        !self.is_valid(index)
    }

    #[inline]
    pub fn slice(&self, range: Range<usize>) -> Self {
        match self {
            Self::Array(a) => Self::Array(a.slice(range)),
            Self::NonNullable | Self::AllValid | Self::AllInvalid => self.clone(),
        }
    }

    pub fn take(&self, indices: &dyn Array) -> VortexResult<Self> {
        match self {
            Self::NonNullable => match indices.validity_mask().boolean_buffer() {
                AllOr::All => {
                    if indices.dtype().is_nullable() {
                        Ok(Self::AllValid)
                    } else {
                        Ok(Self::NonNullable)
                    }
                }
                AllOr::None => Ok(Self::AllInvalid),
                AllOr::Some(buf) => Ok(Validity::from(buf.clone())),
            },
            Self::AllValid => match indices.validity_mask().boolean_buffer() {
                AllOr::All => Ok(Self::AllValid),
                AllOr::None => Ok(Self::AllInvalid),
                AllOr::Some(buf) => Ok(Validity::from(buf.clone())),
            },
            Self::AllInvalid => Ok(Self::AllInvalid),
            Self::Array(is_valid) => {
                let maybe_is_valid = take(is_valid, indices)?;
                // Null indices invalidate that position.
                let is_valid = fill_null(&maybe_is_valid, &Scalar::from(false))?;
                Ok(Self::Array(is_valid))
            }
        }
    }

    /// Keep only the entries for which the mask is true.
    ///
    /// The result has length equal to the number of true values in mask.
    pub fn filter(&self, mask: &Mask) -> VortexResult<Self> {
        // NOTE(ngates): we take the mask as a reference to avoid the caller cloning unnecessarily
        //  if we happen to be NonNullable, AllValid, or AllInvalid.
        match self {
            v @ (Validity::NonNullable | Validity::AllValid | Validity::AllInvalid) => {
                Ok(v.clone())
            }
            Validity::Array(arr) => Ok(Validity::Array(filter(arr, mask)?)),
        }
    }

    /// Set to false any entries for which the mask is true.
    ///
    /// The result is always nullable. The result has the same length as self.
    #[inline]
    pub fn mask(&self, mask: &Mask) -> Self {
        match mask.boolean_buffer() {
            AllOr::All => Validity::AllInvalid,
            AllOr::None => self.clone(),
            AllOr::Some(make_invalid) => match self {
                Validity::NonNullable | Validity::AllValid => {
                    Validity::Array(BoolArray::from(make_invalid.not()).into_array())
                }
                Validity::AllInvalid => Validity::AllInvalid,
                Validity::Array(is_valid) => {
                    let is_valid = is_valid.to_bool();
                    let keep_valid = make_invalid.not();
                    Validity::from(is_valid.boolean_buffer().bitand(&keep_valid))
                }
            },
        }
    }

    #[inline]
    pub fn to_mask(&self, length: usize) -> Mask {
        match self {
            Self::NonNullable | Self::AllValid => Mask::AllTrue(length),
            Self::AllInvalid => Mask::AllFalse(length),
            Self::Array(is_valid) => {
                assert_eq!(
                    is_valid.len(),
                    length,
                    "Validity::Array length must equal to_logical's argument: {}, {}.",
                    is_valid.len(),
                    length,
                );
                is_valid.to_bool().to_mask()
            }
        }
    }

    /// Logically & two Validity values of the same length
    #[inline]
    pub fn and(self, rhs: Validity) -> Validity {
        match (self, rhs) {
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
                let lhs = lhs.to_bool();
                let rhs = rhs.to_bool();

                let lhs = lhs.boolean_buffer();
                let rhs = rhs.boolean_buffer();

                Validity::from(lhs.bitand(rhs))
            }
        }
    }

    pub fn patch(
        self,
        len: usize,
        indices_offset: usize,
        indices: &dyn Array,
        patches: &Validity,
    ) -> Self {
        match (&self, patches) {
            (Validity::NonNullable, Validity::NonNullable) => return Validity::NonNullable,
            (Validity::NonNullable, _) => {
                vortex_panic!("Can't patch a non-nullable validity with nullable validity")
            }
            (_, Validity::NonNullable) => {
                vortex_panic!("Can't patch a nullable validity with non-nullable validity")
            }
            (Validity::AllValid, Validity::AllValid) => return Validity::AllValid,
            (Validity::AllInvalid, Validity::AllInvalid) => return Validity::AllInvalid,
            _ => {}
        };

        let own_nullability = if self == Validity::NonNullable {
            Nullability::NonNullable
        } else {
            Nullability::Nullable
        };

        let source = match self {
            Validity::NonNullable => BoolArray::from(BooleanBuffer::new_set(len)),
            Validity::AllValid => BoolArray::from(BooleanBuffer::new_set(len)),
            Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(len)),
            Validity::Array(a) => a.to_bool(),
        };

        let patch_values = match patches {
            Validity::NonNullable => BoolArray::from(BooleanBuffer::new_set(indices.len())),
            Validity::AllValid => BoolArray::from(BooleanBuffer::new_set(indices.len())),
            Validity::AllInvalid => BoolArray::from(BooleanBuffer::new_unset(indices.len())),
            Validity::Array(a) => a.to_bool(),
        };

        let patches = Patches::new(
            len,
            indices_offset,
            indices.to_array(),
            patch_values.into_array(),
        );

        Self::from_array(source.patch(&patches).into_array(), own_nullability)
    }

    /// Convert into a nullable variant
    #[inline]
    pub fn into_nullable(self) -> Validity {
        match self {
            Self::NonNullable => Self::AllValid,
            Self::AllValid | Self::AllInvalid | Self::Array(_) => self,
        }
    }

    /// Convert into a non-nullable variant
    #[inline]
    pub fn into_non_nullable(self) -> Option<Validity> {
        match self {
            Self::NonNullable => Some(Self::NonNullable),
            Self::AllValid => Some(Self::NonNullable),
            Self::AllInvalid => None,
            Self::Array(is_valid) => {
                is_valid
                    .statistics()
                    .compute_min::<bool>()
                    .vortex_expect("validity array must support min")
                    .then(|| {
                        // min true => all true
                        Self::NonNullable
                    })
            }
        }
    }

    /// Convert into a variant compatible with the given nullability, if possible.
    #[inline]
    pub fn cast_nullability(self, nullability: Nullability) -> VortexResult<Validity> {
        match nullability {
            Nullability::NonNullable => self.into_non_nullable().ok_or_else(|| {
                vortex_err!("Cannot cast array with invalid values to non-nullable type.")
            }),
            Nullability::Nullable => Ok(self.into_nullable()),
        }
    }

    /// Create Validity by copying the given array's validity.
    #[inline]
    pub fn copy_from_array(array: &dyn Array) -> Self {
        Validity::from_mask(array.validity_mask(), array.dtype().nullability())
    }

    /// Create Validity from boolean array with given nullability of the array.
    ///
    /// Note: You want to pass the nullability of parent array and not the nullability of the validity array itself
    ///     as that is always nonnullable
    #[inline]
    fn from_array(value: ArrayRef, nullability: Nullability) -> Self {
        if !matches!(value.dtype(), DType::Bool(Nullability::NonNullable)) {
            vortex_panic!("Expected a non-nullable boolean array")
        }
        match nullability {
            Nullability::NonNullable => Self::NonNullable,
            Nullability::Nullable => Self::Array(value),
        }
    }

    /// Returns the length of the validity array, if it exists.
    #[inline]
    pub fn maybe_len(&self) -> Option<usize> {
        match self {
            Self::NonNullable | Self::AllValid | Self::AllInvalid => None,
            Self::Array(a) => Some(a.len()),
        }
    }

    #[inline]
    pub fn uncompressed_size(&self) -> usize {
        if let Validity::Array(a) = self {
            a.len().div_ceil(8)
        } else {
            0
        }
    }
}

impl PartialEq for Validity {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NonNullable, Self::NonNullable) => true,
            (Self::AllValid, Self::AllValid) => true,
            (Self::AllInvalid, Self::AllInvalid) => true,
            (Self::Array(a), Self::Array(b)) => {
                let a = a.to_bool();
                let b = b.to_bool();
                a.boolean_buffer() == b.boolean_buffer()
            }
            _ => false,
        }
    }
}

impl From<BooleanBuffer> for Validity {
    #[inline]
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
    #[inline]
    fn from(value: NullBuffer) -> Self {
        value.into_inner().into()
    }
}

impl FromIterator<Mask> for Validity {
    #[inline]
    fn from_iter<T: IntoIterator<Item = Mask>>(iter: T) -> Self {
        Validity::from_mask(iter.into_iter().collect(), Nullability::Nullable)
    }
}

impl FromIterator<bool> for Validity {
    #[inline]
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        Validity::from(BooleanBuffer::from_iter(iter))
    }
}

impl From<Nullability> for Validity {
    #[inline]
    fn from(value: Nullability) -> Self {
        match value {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        }
    }
}

impl Validity {
    pub fn from_mask(mask: Mask, nullability: Nullability) -> Self {
        assert!(
            nullability == Nullability::Nullable || matches!(mask, Mask::AllTrue(_)),
            "NonNullable validity must be AllValid",
        );
        match mask {
            Mask::AllTrue(_) => match nullability {
                Nullability::NonNullable => Validity::NonNullable,
                Nullability::Nullable => Validity::AllValid,
            },
            Mask::AllFalse(_) => Validity::AllInvalid,
            Mask::Values(values) => Validity::Array(values.into_array()),
        }
    }
}

impl IntoArray for Mask {
    #[inline]
    fn into_array(self) -> ArrayRef {
        match self {
            Self::AllTrue(len) => ConstantArray::new(true, len).into_array(),
            Self::AllFalse(len) => ConstantArray::new(false, len).into_array(),
            Self::Values(a) => a.into_array(),
        }
    }
}

impl IntoArray for &MaskValues {
    #[inline]
    fn into_array(self) -> ArrayRef {
        BoolArray::from_bool_buffer(self.boolean_buffer().clone(), Validity::NonNullable)
            .into_array()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::Nullability;
    use vortex_mask::Mask;

    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::validity::Validity;
    use crate::{ArrayRef, IntoArray};

    #[rstest]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllValid, Validity::AllValid)]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllInvalid, Validity::Array(BoolArray::from_iter([true, true, false, true, false]).into_array())
    )]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([true, true, true, true, false]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllValid, Validity::Array(BoolArray::from_iter([false, false, true, false, true]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllInvalid, Validity::AllInvalid)]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::Array(BoolArray::from_iter([true, false]).into_array()), Validity::Array(BoolArray::from_iter([false, false, true, false, false]).into_array())
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
        #[case] positions: &[u64],
        #[case] patches: Validity,
        #[case] expected: Validity,
    ) {
        let indices =
            PrimitiveArray::new(Buffer::copy_from(positions), Validity::NonNullable).into_array();
        assert_eq!(validity.patch(len, 0, &indices, &patches), expected);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_patch() {
        Validity::NonNullable.patch(2, 0, &buffer![4].into_array(), &Validity::AllInvalid);
    }

    #[test]
    #[should_panic]
    fn into_validity_nullable() {
        Validity::from_mask(Mask::AllFalse(10), Nullability::NonNullable);
    }

    #[test]
    #[should_panic]
    fn into_validity_nullable_array() {
        Validity::from_mask(Mask::from_iter(vec![true, false]), Nullability::NonNullable);
    }

    #[rstest]
    #[case(Validity::AllValid, PrimitiveArray::new(buffer![0, 1], Validity::from_iter(vec![true, false])).into_array(), Validity::from_iter(vec![true, false]))]
    #[case(Validity::AllValid, buffer![0, 1].into_array(), Validity::AllValid)]
    #[case(Validity::AllValid, PrimitiveArray::new(buffer![0, 1], Validity::AllInvalid).into_array(), Validity::AllInvalid)]
    #[case(Validity::NonNullable, PrimitiveArray::new(buffer![0, 1], Validity::from_iter(vec![true, false])).into_array(), Validity::from_iter(vec![true, false]))]
    #[case(Validity::NonNullable, buffer![0, 1].into_array(), Validity::NonNullable)]
    #[case(Validity::NonNullable, PrimitiveArray::new(buffer![0, 1], Validity::AllInvalid).into_array(), Validity::AllInvalid)]
    fn validity_take(
        #[case] validity: Validity,
        #[case] indices: ArrayRef,
        #[case] expected: Validity,
    ) {
        assert_eq!(validity.take(&indices).unwrap(), expected);
    }
}
