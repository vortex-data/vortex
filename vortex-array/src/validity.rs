// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array validity and nullability behavior, used by arrays and compute functions.

use std::fmt::Debug;
use std::ops::Range;

use itertools::Itertools as _;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::optimizer::ArrayOptimizer;
use crate::patches::Patches;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

/// Validity information for an array
#[derive(Clone)]
pub enum Validity {
    /// Items *can't* be null
    NonNullable,
    /// All items are valid
    AllValid,
    /// All items are null
    AllInvalid,
    /// The validity of each position in the array is determined by a boolean array.
    ///
    /// True values are valid, false values are invalid ("null").
    Array(ArrayRef),
}

impl Debug for Validity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonNullable => write!(f, "NonNullable"),
            Self::AllValid => write!(f, "AllValid"),
            Self::AllInvalid => write!(f, "AllInvalid"),
            Self::Array(arr) => write!(f, "SomeValid({})", arr.display_values()),
        }
    }
}

impl Validity {
    /// Make a step towards canonicalising validity if necessary
    pub fn execute(self, ctx: &mut ExecutionCtx) -> VortexResult<Validity> {
        match self {
            v @ Validity::NonNullable | v @ Validity::AllValid | v @ Validity::AllInvalid => Ok(v),
            Validity::Array(a) => Ok(Validity::Array(a.execute::<Canonical>(ctx)?.into_array())),
        }
    }
}

impl Validity {
    /// The [`DType`] of the underlying validity array (if it exists).
    pub const DTYPE: DType = DType::Bool(Nullability::NonNullable);

    /// Convert the validity to an array representation.
    pub fn to_array(&self, len: usize) -> ArrayRef {
        match self {
            Self::NonNullable | Self::AllValid => ConstantArray::new(true, len).into_array(),
            Self::AllInvalid => ConstantArray::new(false, len).into_array(),
            Self::Array(a) => a.clone(),
        }
    }

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

    /// Returns `true` if this validity guarantees no null values, i.e. it is either
    /// [`Validity::NonNullable`] or [`Validity::AllValid`].
    #[inline]
    pub fn no_nulls(&self) -> bool {
        matches!(self, Self::NonNullable | Self::AllValid)
    }

    /// The union nullability and validity.
    #[inline]
    pub fn union_nullability(self, nullability: Nullability) -> Self {
        match nullability {
            Nullability::NonNullable => self,
            Nullability::Nullable => self.into_nullable(),
        }
    }

    /// Returns whether the `index` item is valid, using `ctx` to execute the validity array.
    #[inline]
    pub fn execute_is_valid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(match self {
            Self::NonNullable | Self::AllValid => true,
            Self::AllInvalid => false,
            Self::Array(a) => a
                .execute_scalar(index, ctx)?
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("validity value at index {index} is null"))?,
        })
    }

    /// Returns whether the `index` item is null, using `ctx` to execute the validity array.
    #[inline]
    pub fn execute_is_null(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(!self.execute_is_valid(index, ctx)?)
    }

    /// Returns whether the `index` item is valid.
    #[deprecated(note = "use `execute_is_valid` with an explicit `ExecutionCtx`")]
    #[inline]
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.execute_is_valid(index, &mut LEGACY_SESSION.create_execution_ctx())
    }

    /// Returns whether the `index` item is null.
    #[deprecated(note = "use `execute_is_null` with an explicit `ExecutionCtx`")]
    #[inline]
    pub fn is_null(&self, index: usize) -> VortexResult<bool> {
        self.execute_is_null(index, &mut LEGACY_SESSION.create_execution_ctx())
    }

    #[inline]
    pub fn slice(&self, range: Range<usize>) -> VortexResult<Self> {
        match self {
            Self::Array(a) => Ok(Self::Array(a.slice(range)?)),
            Self::NonNullable | Self::AllValid | Self::AllInvalid => Ok(self.clone()),
        }
    }

    pub fn take(&self, indices: &ArrayRef) -> VortexResult<Self> {
        match self {
            Self::NonNullable => indices.validity(),
            Self::AllValid => Ok(match indices.validity()? {
                Self::NonNullable => Self::AllValid,
                v => v,
            }),
            Self::AllInvalid => Ok(Self::AllInvalid),
            Self::Array(is_valid) => {
                let maybe_is_valid = is_valid.take(indices.clone())?;
                // Null indices invalidate that position.
                let is_valid = maybe_is_valid.fill_null(Scalar::from(false))?;
                Ok(Self::Array(is_valid))
            }
        }
    }

    // Invert the validity
    pub fn not(&self) -> VortexResult<Self> {
        match self {
            Validity::NonNullable => Ok(Validity::NonNullable),
            Validity::AllValid => Ok(Validity::AllInvalid),
            Validity::AllInvalid => Ok(Validity::AllValid),
            Validity::Array(arr) => Ok(Validity::Array(arr.not()?)),
        }
    }

    /// Lazily filters a [`Validity`] with a selection mask, which keeps only the entries for which
    /// the mask is true.
    ///
    /// The result has length equal to the number of true values in mask.
    ///
    /// If the validity is a [`Validity::Array`], then this lazily wraps it in a `FilterArray`
    /// instead of eagerly filtering the values immediately.
    pub fn filter(&self, mask: &Mask) -> VortexResult<Self> {
        // NOTE(ngates): we take the mask as a reference to avoid the caller cloning unnecessarily
        //  if we happen to be NonNullable, AllValid, or AllInvalid.
        match self {
            v @ (Validity::NonNullable | Validity::AllValid | Validity::AllInvalid) => {
                Ok(v.clone())
            }
            Validity::Array(arr) => Ok(Validity::Array(arr.filter(mask.clone())?)),
        }
    }

    /// Converts this validity into a [`Mask`] of the given length.
    ///
    /// Valid elements are `true` and invalid elements are `false`.
    #[deprecated(note = "Use execute_mask")]
    pub fn to_mask(&self, length: usize, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        match self {
            Self::NonNullable | Self::AllValid => Ok(Mask::new_true(length)),
            Self::AllInvalid => Ok(Mask::new_false(length)),
            Self::Array(arr) => arr.clone().execute::<Mask>(ctx),
        }
    }

    pub fn execute_mask(&self, length: usize, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        match self {
            Self::NonNullable | Self::AllValid => Ok(Mask::AllTrue(length)),
            Self::AllInvalid => Ok(Mask::AllFalse(length)),
            Self::Array(arr) => {
                assert_eq!(
                    arr.len(),
                    length,
                    "Validity::Array length must equal to_logical's argument: {}, {}.",
                    arr.len(),
                    length,
                );
                // TODO(ngates): I'm not sure execution should take arrays by ownership.
                //  If so we should fix call sites to clone and this function takes self.
                arr.clone().execute::<Mask>(ctx)
            }
        }
    }

    /// Compare two Validity values of the same length by executing them into masks if necessary.
    pub fn mask_eq(&self, other: &Validity, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match (self, other) {
            (Validity::NonNullable, Validity::NonNullable) => Ok(true),
            (Validity::AllValid, Validity::AllValid) => Ok(true),
            (Validity::AllInvalid, Validity::AllInvalid) => Ok(true),
            (Validity::Array(a), Validity::Array(b)) => {
                let a = a.clone().execute::<Mask>(ctx)?;
                let b = b.clone().execute::<Mask>(ctx)?;
                Ok(a == b)
            }
            _ => Ok(false),
        }
    }

    /// Logically & two Validity values of the same length
    #[inline]
    pub fn and(self, rhs: Validity) -> VortexResult<Validity> {
        Ok(match (self, rhs) {
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
            (Validity::Array(lhs), Validity::Array(rhs)) => Validity::Array(
                Binary
                    .try_new_array(lhs.len(), Operator::And, [lhs, rhs])?
                    .optimize()?,
            ),
        })
    }

    pub fn patch(
        self,
        len: usize,
        indices_offset: usize,
        indices: &ArrayRef,
        patches: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        match (&self, patches) {
            (Validity::NonNullable, Validity::NonNullable) => return Ok(Validity::NonNullable),
            (Validity::NonNullable, _) => {
                vortex_bail!("Can't patch a non-nullable validity with nullable validity")
            }
            (_, Validity::NonNullable) => {
                vortex_bail!("Can't patch a nullable validity with non-nullable validity")
            }
            (Validity::AllValid, Validity::AllValid) => return Ok(Validity::AllValid),
            (Validity::AllInvalid, Validity::AllInvalid) => return Ok(Validity::AllInvalid),
            _ => {}
        };

        if matches!(self, Validity::NonNullable) {
            return Ok(Self::NonNullable);
        }

        // From here on we know that the validity is nullable
        let source = match self {
            Validity::NonNullable => BoolArray::from(BitBuffer::new_set(len)),
            Validity::AllValid => BoolArray::from(BitBuffer::new_set(len)),
            Validity::AllInvalid => BoolArray::from(BitBuffer::new_unset(len)),
            Validity::Array(a) => a.execute::<BoolArray>(ctx)?,
        };

        let patch_values = match patches {
            Validity::NonNullable => BoolArray::from(BitBuffer::new_set(indices.len())),
            Validity::AllValid => BoolArray::from(BitBuffer::new_set(indices.len())),
            Validity::AllInvalid => BoolArray::from(BitBuffer::new_unset(indices.len())),
            Validity::Array(a) => a.clone().execute::<BoolArray>(ctx)?,
        };

        let patches = Patches::new(
            len,
            indices_offset,
            indices.clone(),
            patch_values.into_array(),
            // TODO(0ax1): chunk offsets
            None,
        )?;

        Ok(Self::Array(source.patch(&patches, ctx)?.into_array()))
    }

    /// Convert into a nullable variant.
    #[inline]
    pub fn into_nullable(self) -> Validity {
        match self {
            Self::NonNullable => Self::AllValid,
            Self::AllValid | Self::AllInvalid | Self::Array(_) => self,
        }
    }

    /// Convert into a non-nullable variant, computing statistics if necessary.
    ///
    /// Returns `None` when the array contains invalid values (so the cast cannot be performed),
    /// either because it is [`Validity::AllInvalid`] or because the validity array's minimum is
    /// `false`.
    #[inline]
    pub fn into_non_nullable(self, len: usize, ctx: &mut ExecutionCtx) -> Option<Validity> {
        match self {
            _ if len == 0 => Some(Validity::NonNullable),
            Self::NonNullable => Some(Self::NonNullable),
            Self::AllValid => Some(Self::NonNullable),
            Self::AllInvalid => None,
            Self::Array(is_valid) => {
                is_valid
                    .statistics()
                    .compute_min::<bool>(ctx)
                    .vortex_expect("validity array must support min")
                    .then(|| {
                        // min true => all true
                        Self::NonNullable
                    })
            }
        }
    }

    /// Convert into a non-nullable variant without running execution.
    ///
    /// This is the cheap counterpart to [`Self::into_non_nullable`]: it inspects already-computed
    /// statistics rather than triggering execution.
    ///
    /// Return values:
    /// - `Ok(Some(NonNullable))` — the cast is provably safe.
    /// - `Ok(None)` — We need to perform compute to determine whether cast is valid. Callers should fall back to [`Self::into_non_nullable`], typically by
    ///   returning `Ok(None)` from a `CastReduce` rule so the corresponding `CastKernel` runs.
    /// - `Err(_)` — we know the cast must fail (e.g. [`Validity::AllInvalid`]).
    #[inline]
    pub fn trivial_into_non_nullable(self, len: usize) -> VortexResult<Option<Validity>> {
        match self {
            _ if len == 0 => Ok(Some(Validity::NonNullable)),
            Self::NonNullable => Ok(Some(Self::NonNullable)),
            Self::AllValid => Ok(Some(Self::NonNullable)),
            Self::AllInvalid => {
                Err(vortex_err!(InvalidArgument: "Cannot cast AllInvalid to NonNullable"))
            }
            Self::Array(_) => Ok(None),
        }
    }

    /// Convert into a variant compatible with the given nullability.
    ///
    /// This is the execution-time half of the nullability-cast pair. It is paired with
    /// [`Self::trivially_cast_nullability`], which is used by `CastReduce` rules. The pattern is:
    ///
    /// - **`CastReduce` rules** (metadata-only rewrites in the optimizer) call
    ///   [`Self::trivially_cast_nullability`]. If it returns `Ok(None)`, the rule returns `Ok(None)`
    ///   and the cast is deferred to execution.
    /// - **`CastKernel` impls** (executed via [`ExecuteParentKernel`]) call this method, which
    ///   may run the underlying validity array to compute statistics.
    ///
    /// Returns `Err` when nullability cannot be cast (for example, casting to non-nullable while
    /// invalid values are present).
    ///
    /// [`ExecuteParentKernel`]: crate::kernel::ExecuteParentKernel
    #[inline]
    pub fn cast_nullability(
        self,
        nullability: Nullability,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Validity> {
        match nullability {
            Nullability::NonNullable => self.into_non_nullable(len, ctx).ok_or_else(|| {
                vortex_err!(InvalidArgument: "Cannot cast array with invalid values to non-nullable type.")
            }),
            Nullability::Nullable => Ok(self.into_nullable()),
        }
    }

    /// Best-effort, non-executing variant of [`Self::cast_nullability`].
    ///
    /// Use this from `CastReduce` rules — they run inside the optimizer where execution is not
    /// available. The pairing with [`Self::cast_nullability`] is symmetric: every encoding that
    /// implements `CastReduce` and inspects validity should also implement `CastKernel` so that
    /// the harder cases (where statistics are not yet cached) can still be handled at execution
    /// time.
    ///
    /// Return values:
    /// - `Ok(Some(_))` — the cast is provably safe and the new [`Validity`] is returned.
    /// - `Ok(None)` — the cast cannot be reduced cheaply (the `CastKernel` should be tried via
    ///   [`Self::cast_nullability`]).
    /// - `Err(_)` — the cast is provably impossible.
    ///
    /// Typical usage inside a `CastReduce`:
    ///
    /// ```ignore
    /// let Some(new_validity) = array
    ///     .validity()?
    ///     .trivial_cast_nullability(dtype.nullability(), array.len())?
    /// else {
    ///     return Ok(None);
    /// };
    /// ```
    #[inline]
    pub fn trivially_cast_nullability(
        self,
        nullability: Nullability,
        len: usize,
    ) -> VortexResult<Option<Validity>> {
        match nullability {
            Nullability::NonNullable => self.trivial_into_non_nullable(len),
            Nullability::Nullable => Ok(Some(self.into_nullable())),
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
}

impl From<BitBuffer> for Validity {
    #[inline]
    fn from(value: BitBuffer) -> Self {
        let true_count = value.true_count();
        if true_count == value.len() {
            Self::AllValid
        } else if true_count == 0 {
            Self::AllInvalid
        } else {
            Self::Array(BoolArray::from(value).into_array())
        }
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
        Validity::from(BitBuffer::from_iter(iter))
    }
}

impl From<Nullability> for Validity {
    #[inline]
    fn from(value: Nullability) -> Self {
        Validity::from(&value)
    }
}

impl From<&Nullability> for Validity {
    #[inline]
    fn from(value: &Nullability) -> Self {
        match *value {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => Validity::AllValid,
        }
    }
}

impl Validity {
    /// Concatenate one or more validities together.
    ///
    /// Returns None if the vector is empty.
    pub fn concat(validities: Vec<(Validity, usize)>) -> Option<Self> {
        let mut validity_kinds = validities
            .iter()
            .map(|(v, _)| std::mem::discriminant(v))
            .unique();
        let validity_kind = validity_kinds.next()?;
        if validity_kinds.next().is_none() {
            // If there is only one kind of validity and its not Validity::Array, avoid constructing
            // a Validity::Array.
            if validity_kind == std::mem::discriminant(&Validity::AllValid) {
                return Some(Validity::AllValid);
            }
            if validity_kind == std::mem::discriminant(&Validity::AllInvalid) {
                return Some(Validity::AllInvalid);
            }
            if validity_kind == std::mem::discriminant(&Validity::NonNullable) {
                return Some(Validity::NonNullable);
            }
        }

        Some(Validity::Array(
            unsafe {
                ChunkedArray::new_unchecked(
                    validities
                        .into_iter()
                        .map(|(v, len)| v.to_array(len))
                        .collect(),
                    DType::Bool(Nullability::NonNullable),
                )
            }
            .into_array(),
        ))
    }
}

impl Validity {
    pub fn from_bit_buffer(buffer: BitBuffer, nullability: Nullability) -> Self {
        if buffer.true_count() == buffer.len() {
            nullability.into()
        } else if buffer.true_count() == 0 {
            Validity::AllInvalid
        } else {
            Validity::Array(BoolArray::new(buffer, Validity::NonNullable).into_array())
        }
    }

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
        BoolArray::new(self.bit_buffer().clone(), Validity::NonNullable).into_array()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
    use crate::validity::BoolArray;
    use crate::validity::Validity;

    #[rstest]
    #[case(Validity::AllValid, 5, &[2, 4], Validity::AllValid, Validity::AllValid)]
    #[case(
        Validity::AllValid,
        5,
        &[2, 4],
        Validity::AllInvalid,
        Validity::Array(BoolArray::from_iter([true, true, false, true, false]).into_array())
    )]
    #[case(
        Validity::AllValid,
        5,
        &[2, 4],
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        Validity::Array(BoolArray::from_iter([true, true, true, true, false]).into_array())
    )]
    #[case(
        Validity::AllInvalid,
        5,
        &[2, 4],
        Validity::AllValid,
        Validity::Array(BoolArray::from_iter([false, false, true, false, true]).into_array())
    )]
    #[case(Validity::AllInvalid, 5, &[2, 4], Validity::AllInvalid, Validity::AllInvalid)]
    #[case(
        Validity::AllInvalid,
        5,
        &[2, 4],
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        Validity::Array(BoolArray::from_iter([false, false, true, false, false]).into_array())
    )]
    #[case(
        Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()),
        5,
        &[2, 4],
        Validity::AllValid,
        Validity::Array(BoolArray::from_iter([false, true, true, true, true]).into_array())
    )]
    #[case(
        Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()),
        5,
        &[2, 4],
        Validity::AllInvalid,
        Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array())
    )]
    #[case(
        Validity::Array(BoolArray::from_iter([false, true, false, true, false]).into_array()),
        5,
        &[2, 4],
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
        Validity::Array(BoolArray::from_iter([false, true, true, true, false]).into_array())
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

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert!(
            validity
                .patch(len, 0, &indices, &patches, &mut ctx,)
                .unwrap()
                .mask_eq(&expected, &mut ctx)
                .unwrap()
        );
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_patch() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        Validity::NonNullable
            .patch(
                2,
                0,
                &buffer![4].into_array(),
                &Validity::AllInvalid,
                &mut ctx,
            )
            .unwrap();
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
    #[case(
        Validity::AllValid,
        PrimitiveArray::new(buffer![0, 1], Validity::from_iter(vec![true, false])).into_array(),
        Validity::from_iter(vec![true, false])
    )]
    #[case(Validity::AllValid, buffer![0, 1].into_array(), Validity::AllValid)]
    #[case(
        Validity::AllValid,
        PrimitiveArray::new(buffer![0, 1], Validity::AllInvalid).into_array(),
        Validity::AllInvalid
    )]
    #[case(
        Validity::NonNullable,
        PrimitiveArray::new(buffer![0, 1], Validity::from_iter(vec![true, false])).into_array(),
        Validity::from_iter(vec![true, false])
    )]
    #[case(Validity::NonNullable, buffer![0, 1].into_array(), Validity::NonNullable)]
    #[case(
        Validity::NonNullable,
        PrimitiveArray::new(buffer![0, 1], Validity::AllInvalid).into_array(),
        Validity::AllInvalid
    )]
    fn validity_take(
        #[case] validity: Validity,
        #[case] indices: ArrayRef,
        #[case] expected: Validity,
    ) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(
            validity
                .take(&indices)
                .unwrap()
                .mask_eq(&expected, &mut ctx)
                .unwrap()
        );
    }
}
