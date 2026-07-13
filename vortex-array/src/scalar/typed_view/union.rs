// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions and implementations of [`UnionScalar`] and [`UnionValue`].

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::UnionVariants;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// The present value stored by a union scalar.
///
/// A null union is represented by the enclosing [`Scalar`]'s value being `None`. The selected
/// child's raw value is optional so a present union whose selected child is null remains distinct
/// from an outer null union. Its dtype is determined by `type_id` and the enclosing
/// [`DType::Union`].
#[derive(Debug, Clone)]
pub struct UnionValue {
    /// The type ID selecting a variant in the enclosing [`DType::Union`].
    type_id: u8,
    /// The selected variant's raw value, or [`None`] if the selected child is null.
    ///
    /// This is boxed to break the recursive layout between [`Scalar`], [`ScalarValue`], and
    /// [`UnionValue`].
    value: Option<Box<ScalarValue>>,
}

impl PartialEq for UnionValue {
    fn eq(&self, other: &Self) -> bool {
        if self.value.is_none() && other.value.is_none() {
            return true;
        }

        self.type_id == other.type_id && self.value == other.value
    }
}

impl Eq for UnionValue {}

impl Hash for UnionValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let is_null = self.value.is_none();
        is_null.hash(state);
        if !is_null {
            self.type_id.hash(state);
            self.value.hash(state);
        }
    }
}

impl UnionValue {
    pub(crate) fn new(type_id: u8, value: Option<ScalarValue>) -> Self {
        Self {
            type_id,
            value: value.map(Box::new),
        }
    }

    /// Returns the type ID selecting the union variant.
    #[inline]
    pub fn type_id(&self) -> u8 {
        self.type_id
    }

    /// Returns the selected variant's raw value, or [`None`] if the selected child is null.
    #[inline]
    pub fn value(&self) -> Option<&ScalarValue> {
        self.value.as_deref()
    }
}

/// A typed view into a [`DType::Union`] scalar.
///
/// A present union scalar carries a type ID and the raw value of the selected variant. An outer
/// null union scalar has neither because outer nullness is represented by the enclosing [`Scalar`].
#[derive(Debug, Clone, Copy)]
pub struct UnionScalar<'a> {
    /// The data type of this scalar.
    dtype: &'a DType,
    /// The selected union value, or [`None`] if the union scalar is null.
    value: Option<&'a UnionValue>,
}

impl Display for UnionScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Some(name) = self.variant_name() else {
            return write!(f, "null");
        };

        write!(f, "{name}(")?;
        match self
            .value
            .vortex_expect("non-null union scalar must have a selected value")
            .value()
        {
            Some(value) => write!(f, "{value}"),
            None => write!(f, "null"),
        }?;
        write!(f, ")")
    }
}

impl PartialEq for UnionScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
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
        vortex_ensure!(dtype.is_union(), "Expected union scalar, found {dtype}");

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
        if !dtype.is_union() {
            vortex_panic!("Expected union scalar, found {dtype}")
        }

        Self {
            dtype,
            value: value.map(ScalarValue::as_union),
        }
    }

    /// Returns the data type of this union scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the variants of this union scalar.
    #[inline]
    pub fn variants(&self) -> &'a UnionVariants {
        self.dtype
            .as_union_variants_opt()
            .vortex_expect("UnionScalar always has union dtype")
    }

    /// Returns the outer nullability of this union scalar.
    #[inline]
    pub fn nullability(&self) -> Nullability {
        self.dtype.nullability()
    }

    /// Returns true if this union scalar is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.value.is_none()
    }

    /// Returns the selected type ID, or `None` if this scalar is null.
    #[inline]
    pub fn type_id(&self) -> Option<u8> {
        Some(self.value?.type_id())
    }

    /// Returns the selected variant's child index, or `None` if this scalar is null.
    pub fn child_index(&self) -> Option<usize> {
        self.variants().tag_to_child_index(self.type_id()?)
    }

    /// Returns the selected variant's name, or `None` if this scalar is null.
    pub fn variant_name(&self) -> Option<&'a FieldName> {
        self.variants().names().get(self.child_index()?)
    }

    /// Returns the selected variant's dtype, or `None` if this scalar is null.
    pub fn variant_dtype(&self) -> Option<DType> {
        self.variants().variant_by_index(self.child_index()?)
    }

    /// Returns the selected child scalar, or [`None`] if the outer union scalar is null.
    ///
    /// A selected null child is returned as a present, null [`Scalar`].
    pub fn value(&self) -> Option<Scalar> {
        let union_value = self.value?;
        let child_dtype = self
            .variant_dtype()
            .vortex_expect("validated union type ID must select a child dtype");

        // SAFETY: Construction of this `UnionScalar` guarantees that its type ID resolves to this
        // child dtype and that the raw child value recursively validates against it.
        Some(unsafe { Scalar::new_unchecked(child_dtype, union_value.value().cloned()) })
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
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
        let child = Scalar::primitive(42_i32, Nullability::Nullable);
        let scalar = Scalar::union(variants()?, 5, child.clone(), Nullability::Nullable)?;

        let union = scalar.as_union();
        assert!(!union.is_null());
        assert_eq!(union.type_id(), Some(5));
        assert_eq!(union.child_index(), Some(0));
        assert_eq!(union.variant_name().map(AsRef::as_ref), Some("int"));
        assert_eq!(union.nullability(), Nullability::Nullable);
        assert_eq!(union.value(), Some(child));
        assert_eq!(
            scalar
                .value()
                .vortex_expect("union must have an outer value")
                .as_union()
                .value(),
            Some(&ScalarValue::Primitive(42_i32.into()))
        );

        Ok(())
    }

    #[test]
    fn inner_null_is_distinct_from_outer_null() -> VortexResult<()> {
        let scalar = Scalar::union(
            variants()?,
            5,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::Nullable,
        )?;

        let union = scalar.as_union();
        assert!(!union.is_null());
        assert_eq!(union.type_id(), Some(5));
        assert!(union.value().is_some_and(|scalar| scalar.is_null()));
        assert!(
            scalar
                .value()
                .vortex_expect("union must have an outer value")
                .as_union()
                .value()
                .is_none()
        );

        let outer_null = Scalar::null(DType::Union(variants()?, Nullability::Nullable));
        let outer_union = outer_null.as_union();
        assert!(outer_union.is_null());
        assert_eq!(outer_union.type_id(), None);
        assert_eq!(outer_union.value(), None);

        Ok(())
    }

    #[test]
    fn child_is_validated() -> VortexResult<()> {
        let variants = variants()?;
        let child = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));

        assert!(Scalar::union(variants.clone(), 7, child, Nullability::NonNullable).is_err());

        let wrong_child = Scalar::null(DType::Utf8(Nullability::Nullable));

        assert!(Scalar::union(variants, 5, wrong_child, Nullability::NonNullable).is_err());

        Ok(())
    }

    #[test]
    fn try_new_rejects_non_union_dtype() {
        assert!(
            UnionScalar::try_new(&DType::Bool(Nullability::Nullable), None).is_err(),
            "non-union dtypes must return an error"
        );
    }

    #[test]
    fn try_new_validates_type_id_and_selected_value() -> VortexResult<()> {
        let dtype = DType::Union(variants()?, Nullability::Nullable);
        let non_nullable_dtype = DType::Union(
            UnionVariants::try_new(
                ["int"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
                vec![5],
            )?,
            Nullability::NonNullable,
        );

        assert!(UnionScalar::try_new(&non_nullable_dtype, None).is_err());

        let unknown_type_id = ScalarValue::Union(UnionValue::new(
            7,
            Scalar::primitive(42_i32, Nullability::Nullable).into_value(),
        ));

        assert!(UnionScalar::try_new(&dtype, Some(&unknown_type_id)).is_err());

        let wrong_value = ScalarValue::Union(UnionValue::new(
            5,
            Scalar::utf8("wrong", Nullability::NonNullable).into_value(),
        ));

        assert!(UnionScalar::try_new(&dtype, Some(&wrong_value)).is_err());

        Ok(())
    }
}
