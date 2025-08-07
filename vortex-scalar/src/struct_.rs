// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::{DType, FieldName, FieldNames, StructFields};
use vortex_error::{
    VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic,
};

use crate::{InnerScalarValue, Scalar, ScalarValue};

/// A scalar value representing a struct with named fields.
///
/// This type provides a view into a struct scalar value, which can contain
/// named fields with different types, or be null.
pub struct StructScalar<'a> {
    dtype: &'a DType,
    fields: Option<&'a Arc<[ScalarValue]>>,
}

impl Display for StructScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.fields {
            None => write!(f, "null"),
            Some(fields) => {
                write!(f, "{{")?;
                let formatted_fields = self
                    .names()
                    .iter()
                    .zip_eq(self.struct_fields().fields())
                    .zip_eq(fields.iter())
                    .map(|((name, dtype), value)| {
                        let val = Scalar::new(dtype, value.clone());
                        format!("{name}: {val}")
                    })
                    .format(", ");
                write!(f, "{formatted_fields}")?;
                write!(f, "}}")
            }
        }
    }
}

impl PartialEq for StructScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return false;
        }
        self.fields() == other.fields()
    }
}

impl Eq for StructScalar<'_> {}

/// Ord is not implemented since it's undefined for different field DTypes
impl PartialOrd for StructScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }
        self.fields().partial_cmp(&other.fields())
    }
}

impl Hash for StructScalar<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.fields().hash(state);
    }
}

impl<'a> StructScalar<'a> {
    pub(crate) fn try_new(dtype: &'a DType, value: &'a ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Struct(..)) {
            vortex_bail!("Expected struct scalar, found {}", dtype)
        }
        Ok(Self {
            dtype,
            fields: value.as_list()?,
        })
    }

    /// Returns the data type of this struct scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the struct field definitions.
    #[inline]
    pub fn struct_fields(&self) -> &StructFields {
        self.dtype
            .as_struct()
            .vortex_expect("StructScalar always has struct dtype")
    }

    /// Returns the field names of the struct.
    pub fn names(&self) -> &FieldNames {
        self.struct_fields().names()
    }

    /// Returns true if the struct is null.
    pub fn is_null(&self) -> bool {
        self.fields.is_none()
    }

    /// Returns the field with the given name as a scalar.
    ///
    /// Returns None if the field doesn't exist.
    pub fn field(&self, name: impl AsRef<str>) -> Option<Scalar> {
        let idx = self.struct_fields().find(name)?;
        self.field_by_idx(idx)
    }

    /// Returns the field at the given index as a scalar.
    ///
    /// Returns None if the index is out of bounds.
    ///
    /// # Panics
    ///
    /// Panics if the struct is null.
    pub fn field_by_idx(&self, idx: usize) -> Option<Scalar> {
        let fields = self
            .fields
            .vortex_expect("Can't take field out of null struct scalar");
        Some(Scalar::new(
            self.struct_fields().field_by_index(idx)?,
            fields[idx].clone(),
        ))
    }

    /// Returns the fields of the struct scalar, or None if the scalar is null.
    pub fn fields(&self) -> Option<Vec<Scalar>> {
        let fields = self.fields?;
        Some(
            (0..fields.len())
                .map(|index| {
                    self.field_by_idx(index)
                        .vortex_expect("never out of bounds")
                })
                .collect::<Vec<_>>(),
        )
    }

    pub(crate) fn field_values(&self) -> Option<&[ScalarValue]> {
        self.fields.map(Arc::deref)
    }

    /// Casts this struct scalar to another struct type.
    ///
    /// # Errors
    ///
    /// Returns an error if the target type is not a struct or if the number of fields don't match.
    pub fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let DType::Struct(st, _) = dtype else {
            vortex_bail!(
                "Cannot cast struct to {}: struct can only be cast to struct",
                dtype
            )
        };
        let own_st = self.struct_fields();

        if st.fields().len() != own_st.fields().len() {
            vortex_bail!(
                "Cannot cast between structs with different number of fields: {} and {}",
                own_st.fields().len(),
                st.fields().len()
            );
        }

        if let Some(fs) = self.field_values() {
            let fields = fs
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    Scalar::new(
                        own_st
                            .field_by_index(i)
                            .vortex_expect("Iterating over scalar fields"),
                        f.clone(),
                    )
                    .cast(
                        &st.field_by_index(i)
                            .vortex_expect("Iterating over scalar fields"),
                    )
                    .map(|s| s.value)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Ok(Scalar::new(
                dtype.clone(),
                ScalarValue(InnerScalarValue::List(fields.into())),
            ))
        } else {
            Ok(Scalar::null(dtype.clone()))
        }
    }

    /// Projects this struct scalar to include only the specified fields.
    ///
    /// # Errors
    ///
    /// Returns an error if the struct cannot be projected or if a field is not found.
    pub fn project(&self, projection: &[FieldName]) -> VortexResult<Scalar> {
        let struct_dtype = self
            .dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("Not a struct dtype"))?;
        let projected_dtype = struct_dtype.project(projection)?;
        let new_fields = if let Some(fs) = self.field_values() {
            ScalarValue(InnerScalarValue::List(
                projection
                    .iter()
                    .map(|name| {
                        struct_dtype
                            .find(name)
                            .vortex_expect("DType has been successfully projected already")
                    })
                    .map(|i| fs[i].clone())
                    .collect(),
            ))
        } else {
            ScalarValue(InnerScalarValue::Null)
        };
        Ok(Scalar::new(
            DType::Struct(projected_dtype, self.dtype().nullability()),
            new_fields,
        ))
    }
}

impl Scalar {
    /// Creates a new struct scalar with the given fields.
    pub fn struct_(dtype: DType, children: Vec<Scalar>) -> Self {
        let DType::Struct(struct_fields, _) = &dtype else {
            vortex_panic!("Expected struct dtype, found {}", dtype);
        };

        let field_dtypes = struct_fields.fields();
        if children.len() != field_dtypes.len() {
            vortex_panic!(
                "Struct has {} fields but {} children were provided",
                field_dtypes.len(),
                children.len()
            );
        }

        for (idx, (child, expected_dtype)) in children.iter().zip(field_dtypes).enumerate() {
            if child.dtype() != &expected_dtype {
                vortex_panic!(
                    "Field {} expected dtype {} but got {}",
                    idx,
                    expected_dtype,
                    child.dtype()
                );
            }
        }

        Self::new(
            dtype,
            ScalarValue(InnerScalarValue::List(
                children
                    .into_iter()
                    .map(|x| x.into_value())
                    .collect_vec()
                    .into(),
            )),
        )
    }
}

impl<'a> TryFrom<&'a Scalar> for StructScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        Self::try_new(value.dtype(), &value.value)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructFields};

    use super::*;

    fn setup_types() -> (DType, DType, DType) {
        let f0_dt = DType::Primitive(I32, Nullability::NonNullable);
        let f1_dt = DType::Utf8(Nullability::NonNullable);

        let dtype = DType::Struct(
            StructFields::new(
                vec!["a".into(), "b".into()].into(),
                vec![f0_dt.clone(), f1_dt.clone()],
            ),
            Nullability::Nullable,
        );

        (f0_dt, f1_dt, dtype)
    }

    #[test]
    #[should_panic]
    fn test_struct_scalar_null() {
        let (_, _, dtype) = setup_types();

        let scalar = Scalar::null(dtype);

        scalar.as_struct().field_by_idx(0).unwrap();
    }

    #[test]
    fn test_struct_scalar_non_null() {
        let (f0_dt, f1_dt, dtype) = setup_types();

        let f0_val = Scalar::primitive::<i32>(1, Nullability::NonNullable);
        let f1_val = Scalar::utf8("hello", Nullability::NonNullable);

        let f0_val_null = Scalar::primitive::<i32>(1, Nullability::Nullable);
        let f1_val_null = Scalar::utf8("hello", Nullability::Nullable);

        let scalar = Scalar::struct_(dtype, vec![f0_val, f1_val]);

        let scalar_f0 = scalar.as_struct().field_by_idx(0);
        assert!(scalar_f0.is_some());
        let scalar_f0 = scalar_f0.unwrap();
        assert_eq!(scalar_f0, f0_val_null);
        assert_eq!(scalar_f0.dtype(), &f0_dt);

        let scalar_f1 = scalar.as_struct().field_by_idx(1);
        assert!(scalar_f1.is_some());
        let scalar_f1 = scalar_f1.unwrap();
        assert_eq!(scalar_f1, f1_val_null);
        assert_eq!(scalar_f1.dtype(), &f1_dt);
    }

    #[test]
    #[should_panic(expected = "Expected struct dtype")]
    fn test_struct_scalar_wrong_dtype() {
        let dtype = DType::Primitive(I32, Nullability::NonNullable);
        let scalar = Scalar::primitive::<i32>(1, Nullability::NonNullable);

        Scalar::struct_(dtype, vec![scalar]);
    }

    #[test]
    #[should_panic(expected = "Struct has 2 fields but 1 children were provided")]
    fn test_struct_scalar_wrong_child_count() {
        let (_, _, dtype) = setup_types();
        let f0_val = Scalar::primitive::<i32>(1, Nullability::NonNullable);

        Scalar::struct_(dtype, vec![f0_val]);
    }

    #[test]
    #[should_panic(expected = "Field 0 expected dtype i32 but got utf8")]
    fn test_struct_scalar_wrong_child_dtype() {
        let (_, _, dtype) = setup_types();
        let f0_val = Scalar::utf8("wrong", Nullability::NonNullable);
        let f1_val = Scalar::utf8("hello", Nullability::NonNullable);

        Scalar::struct_(dtype, vec![f0_val, f1_val]);
    }
}
