// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions and implementations of [`UnionScalar`] and [`UnionValue`].

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::UnionVariants;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// The non-null value stored by a union scalar.
///
/// A null union is represented by the enclosing [`Scalar`]'s value being `None`, rather than by a
/// nested null in this type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnionValue {
    /// The type ID selecting a variant in the enclosing [`DType::Union`].
    type_id: i8,
    /// The selected variant's non-null value, which must be valid for its dtype.
    ///
    /// This is boxed to break the recursive layout between [`ScalarValue`] and [`UnionValue`].
    value: Box<ScalarValue>,
}

impl UnionValue {
    pub(crate) fn new(type_id: i8, value: ScalarValue) -> Self {
        Self {
            type_id,
            value: Box::new(value),
        }
    }

    /// Returns the type ID selecting the union variant.
    #[inline]
    pub fn type_id(&self) -> i8 {
        self.type_id
    }

    /// Returns the selected variant's non-null value.
    #[inline]
    pub fn value(&self) -> &ScalarValue {
        &self.value
    }
}

/// A typed view into a [`DType::Union`] scalar.
///
/// A non-null union scalar carries a type ID and a non-null value of the selected variant. A null
/// union scalar has neither because nullness is represented by the enclosing [`Scalar`].
///
/// The type ID is therefore not preserved across round trips for null union scalars.
#[derive(Debug, Clone, Copy)]
pub struct UnionScalar<'a> {
    /// The variants of the union.
    ///
    /// The enclosing scalar's dtype is always `DType::Union(variants)`, so we hold the unwrapped
    /// variants directly rather than re-matching the dtype on every access.
    variants: &'a UnionVariants,
    /// The selected union value, or [`None`] if the union scalar is null.
    value: Option<&'a UnionValue>,
}

impl Display for UnionScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Some(name) = self.variant_name() else {
            return write!(f, "null");
        };
        let value = self
            .value()
            .vortex_expect("non-null union scalar must have a selected value");

        write!(f, "{name}({value})")
    }
}

impl PartialEq for UnionScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Match the nullability-agnostic equality of `Scalar`: compare the variant schema ignoring
        // per-variant nullability, then the selected value.
        self.variants.names() == other.variants.names()
            && self.variants.type_ids() == other.variants.type_ids()
            && self
                .variants
                .variants()
                .zip(other.variants.variants())
                .all(|(lhs, rhs)| lhs.eq_ignore_nullability(&rhs))
            && self.value == other.value
    }
}

impl Eq for UnionScalar<'_> {}

impl<'a> UnionScalar<'a> {
    /// Attempts to create a union scalar view from a [`DType`] and optional [`ScalarValue`].
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` and `value` do not form a valid union scalar. This includes a
    /// non-union dtype, a null value for a non-nullable union, an unknown type ID, or a value that
    /// is invalid for the selected variant dtype.
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        Scalar::validate(dtype, value)?;

        Ok(Self::new_unchecked(dtype, value))
    }

    /// Creates a union scalar view without validating the scalar value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `dtype` and `value` form a valid union scalar.
    ///
    /// # Panics
    ///
    /// Panics if `dtype` is not a union dtype or a non-null `value` is not a union value.
    pub(crate) fn new_unchecked(dtype: &'a DType, value: Option<&'a ScalarValue>) -> Self {
        let DType::Union(variants) = dtype else {
            vortex_panic!("Expected union scalar, found {dtype}")
        };

        Self {
            variants,
            value: value.map(ScalarValue::as_union),
        }
    }

    /// Returns the variants of this union scalar.
    #[inline]
    pub fn variants(&self) -> &'a UnionVariants {
        self.variants
    }

    /// Returns true if this union scalar is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.value.is_none()
    }

    /// Returns the selected type ID, or `None` if this scalar is null.
    #[inline]
    pub fn type_id(&self) -> Option<i8> {
        Some(self.value?.type_id())
    }

    /// Returns the selected variant's child index, or `None` if this scalar is null.
    pub fn child_index(&self) -> Option<usize> {
        self.variants.tag_to_child_index(self.type_id()?)
    }

    /// Returns the selected variant's name, or `None` if this scalar is null.
    pub fn variant_name(&self) -> Option<&'a FieldName> {
        self.variants.names().get(self.child_index()?)
    }

    /// Returns the selected variant's dtype, or `None` if this scalar is null.
    pub fn variant_dtype(&self) -> Option<DType> {
        self.variants.variant_by_index(self.child_index()?)
    }

    /// Reconstructs the selected child scalar, or returns `None` if this scalar is null.
    pub fn value(&self) -> Option<Scalar> {
        let value = self.value?;
        let dtype = self.variant_dtype()?;

        // SAFETY: Union scalar validation checks the type ID and validates this value against the
        // selected variant dtype.
        Some(unsafe { Scalar::new_unchecked(dtype, Some(value.value().clone())) })
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::UnionScalar;
    use super::UnionValue;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    fn variants() -> VortexResult<UnionVariants> {
        UnionVariants::try_new(
            ["int", "string"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::Nullable),
                DType::Utf8(Nullability::NonNullable),
            ],
            vec![5, 9],
        )
    }

    #[test]
    fn non_null_union_view() -> VortexResult<()> {
        let scalar = Scalar::union(
            variants()?,
            5,
            Scalar::primitive(42_i32, Nullability::Nullable),
        )?;

        let union = scalar.as_union();
        assert!(!union.is_null());
        assert_eq!(union.type_id(), Some(5));
        assert_eq!(union.child_index(), Some(0));
        assert_eq!(union.variant_name().map(AsRef::as_ref), Some("int"));
        assert_eq!(
            union.value(),
            Some(Scalar::primitive(42_i32, Nullability::Nullable))
        );

        Ok(())
    }

    #[test]
    fn null_child_is_normalized_to_outer_null() -> VortexResult<()> {
        let scalar = Scalar::union(
            variants()?,
            5,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        )?;

        let union = scalar.as_union();
        assert!(union.is_null());
        assert_eq!(union.type_id(), None);
        assert_eq!(union.value(), None);

        Ok(())
    }

    #[test]
    fn null_child_is_validated_before_normalization() -> VortexResult<()> {
        let variants = variants()?;
        let child = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));

        assert!(Scalar::union(variants.clone(), 7, child).is_err());

        let wrong_child = Scalar::null(DType::Utf8(Nullability::Nullable));

        assert!(Scalar::union(variants, 5, wrong_child).is_err());

        Ok(())
    }

    #[test]
    fn try_new_validates_type_id_and_selected_value() -> VortexResult<()> {
        let dtype = DType::Union(variants()?);
        let non_nullable_dtype = DType::Union(UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![5],
        )?);

        assert!(UnionScalar::try_new(&non_nullable_dtype, None).is_err());

        let unknown_type_id =
            ScalarValue::Union(UnionValue::new(7, ScalarValue::Primitive(42_i32.into())));

        assert!(UnionScalar::try_new(&dtype, Some(&unknown_type_id)).is_err());

        let wrong_value = ScalarValue::Union(UnionValue::new(5, ScalarValue::Utf8("wrong".into())));

        assert!(UnionScalar::try_new(&dtype, Some(&wrong_value)).is_err());

        Ok(())
    }
}
