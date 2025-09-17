// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
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
#[derive(Debug)]
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

        match (self.fields(), other.fields()) {
            (Some(lhs), Some(rhs)) => lhs.zip(rhs).all(|(l_s, r_s)| l_s == r_s),
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
        }
    }
}

impl Eq for StructScalar<'_> {}

/// Ord is not implemented since it's undefined for different field DTypes
impl PartialOrd for StructScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }

        match (self.fields(), other.fields()) {
            (Some(lhs), Some(rhs)) => {
                for (l_s, r_s) in lhs.zip(rhs) {
                    match l_s.partial_cmp(&r_s)? {
                        Ordering::Equal => continue,
                        Ordering::Less => return Some(Ordering::Less),
                        Ordering::Greater => return Some(Ordering::Greater),
                    }
                }
            }
            (None, None) => return Some(Ordering::Equal),
            (Some(_), None) => return Some(Ordering::Greater),
            (None, Some(_)) => return Some(Ordering::Less),
        }

        Some(Ordering::Equal)
    }
}

impl Hash for StructScalar<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        if let Some(fields) = self.fields() {
            for f in fields {
                f.hash(state);
            }
        }
    }
}

impl<'a> StructScalar<'a> {
    #[inline]
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
            .as_struct_fields_opt()
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
    pub fn fields(&self) -> Option<impl ExactSizeIterator<Item = Scalar>> {
        let fields = self.fields?;
        Some(
            fields
                .iter()
                .zip(self.struct_fields().fields())
                .map(|(v, dtype)| Scalar::new(dtype, v.clone())),
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
                    .map(|s| s.into_value())
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
            .as_struct_fields_opt()
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

        let mut value_children = Vec::with_capacity(children.len());
        value_children.extend(children.into_iter().map(|x| x.into_value()));

        Self::new(
            dtype,
            ScalarValue(InnerScalarValue::List(value_children.into())),
        )
    }
}

impl<'a> TryFrom<&'a Scalar> for StructScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        Self::try_new(value.dtype(), value.value())
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
            StructFields::new(["a", "b"].into(), vec![f0_dt.clone(), f1_dt.clone()]),
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

    #[test]
    fn test_struct_field_by_name() {
        let (_, _, dtype) = setup_types();
        let f0_val = Scalar::primitive::<i32>(42, Nullability::NonNullable);
        let f1_val = Scalar::utf8("world", Nullability::NonNullable);

        let scalar = Scalar::struct_(dtype, vec![f0_val, f1_val]);

        // Get field by name
        let field_a = scalar.as_struct().field("a");
        assert!(field_a.is_some());
        assert_eq!(
            field_a
                .unwrap()
                .as_primitive()
                .typed_value::<i32>()
                .unwrap(),
            42
        );

        let field_b = scalar.as_struct().field("b");
        assert!(field_b.is_some());
        assert_eq!(field_b.unwrap().as_utf8().value().unwrap(), "world".into());

        // Non-existent field
        let field_c = scalar.as_struct().field("c");
        assert!(field_c.is_none());
    }

    #[test]
    fn test_struct_fields() {
        let (_, _, dtype) = setup_types();
        let f0_val = Scalar::primitive::<i32>(100, Nullability::NonNullable);
        let f1_val = Scalar::utf8("test", Nullability::NonNullable);

        let scalar = Scalar::struct_(dtype, vec![f0_val, f1_val]);

        let fields = scalar.as_struct().fields().unwrap().collect::<Vec<_>>();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].as_primitive().typed_value::<i32>().unwrap(), 100);
        assert_eq!(fields[1].as_utf8().value().unwrap(), "test".into());
    }

    #[test]
    fn test_struct_null_fields() {
        let (_, _, dtype) = setup_types();
        let null_scalar = Scalar::null(dtype);

        assert!(null_scalar.as_struct().is_null());
        assert!(null_scalar.as_struct().fields().is_none());
        assert!(null_scalar.as_struct().field_values().is_none());
    }

    #[test]
    fn test_struct_cast_to_struct() {
        // Create source struct
        let source_fields = StructFields::new(
            ["x", "y"].into(),
            vec![
                DType::Primitive(I32, Nullability::NonNullable),
                DType::Primitive(I32, Nullability::NonNullable),
            ],
        );
        let source_dtype = DType::Struct(source_fields, Nullability::NonNullable);

        // Create target struct with different field types
        let target_fields = StructFields::new(
            ["x", "y"].into(),
            vec![
                DType::Primitive(vortex_dtype::PType::I64, Nullability::NonNullable),
                DType::Primitive(vortex_dtype::PType::I64, Nullability::NonNullable),
            ],
        );
        let target_dtype = DType::Struct(target_fields, Nullability::NonNullable);

        let f0 = Scalar::primitive::<i32>(42, Nullability::NonNullable);
        let f1 = Scalar::primitive::<i32>(123, Nullability::NonNullable);
        let source_scalar = Scalar::struct_(source_dtype, vec![f0, f1]);

        // Cast to target type
        let result = source_scalar.as_struct().cast(&target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);

        let fields = result.as_struct().fields().unwrap().collect::<Vec<_>>();
        assert_eq!(fields[0].as_primitive().typed_value::<i64>().unwrap(), 42);
        assert_eq!(fields[1].as_primitive().typed_value::<i64>().unwrap(), 123);
    }

    #[test]
    fn test_struct_cast_mismatched_fields() {
        let source_fields = StructFields::new(
            ["a"].into(),
            vec![DType::Primitive(I32, Nullability::NonNullable)],
        );
        let source_dtype = DType::Struct(source_fields, Nullability::NonNullable);

        let target_fields = StructFields::new(
            ["a", "b"].into(),
            vec![
                DType::Primitive(I32, Nullability::NonNullable),
                DType::Primitive(I32, Nullability::NonNullable),
            ],
        );
        let target_dtype = DType::Struct(target_fields, Nullability::NonNullable);

        let scalar = Scalar::struct_(
            source_dtype,
            vec![Scalar::primitive::<i32>(1, Nullability::NonNullable)],
        );

        let result = scalar.as_struct().cast(&target_dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_struct_cast_to_non_struct() {
        let (_, _, dtype) = setup_types();
        let scalar = Scalar::struct_(
            dtype,
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        let result = scalar
            .as_struct()
            .cast(&DType::Primitive(I32, Nullability::NonNullable));
        assert!(result.is_err());
    }

    #[test]
    fn test_struct_project() {
        let (_, _, dtype) = setup_types();
        let f0_val = Scalar::primitive::<i32>(42, Nullability::NonNullable);
        let f1_val = Scalar::utf8("hello", Nullability::NonNullable);

        let scalar = Scalar::struct_(dtype, vec![f0_val, f1_val]);

        // Project to only field "b"
        let projected = scalar.as_struct().project(&["b".into()]).unwrap();
        let projected_struct = projected.as_struct();

        assert_eq!(projected_struct.names().len(), 1);
        assert_eq!(projected_struct.names()[0].as_ref(), "b");

        let fields = projected_struct.fields().unwrap().collect::<Vec<_>>();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].as_utf8().value().unwrap().as_str(), "hello");
    }

    #[test]
    fn test_struct_project_null() {
        let (_, _, dtype) = setup_types();
        let null_scalar = Scalar::null(dtype);

        let projected = null_scalar.as_struct().project(&["a".into()]).unwrap();
        assert!(projected.as_struct().is_null());
    }

    #[test]
    fn test_struct_equality() {
        let (_, _, dtype) = setup_types();

        let scalar1 = Scalar::struct_(
            dtype.clone(),
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        let scalar2 = Scalar::struct_(
            dtype.clone(),
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        let scalar3 = Scalar::struct_(
            dtype,
            vec![
                Scalar::primitive::<i32>(2, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        assert_eq!(scalar1.as_struct(), scalar2.as_struct());
        assert_ne!(scalar1.as_struct(), scalar3.as_struct());
    }

    #[test]
    fn test_struct_partial_ord() {
        let (_, _, dtype) = setup_types();

        let scalar1 = Scalar::struct_(
            dtype.clone(),
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("a", Nullability::NonNullable),
            ],
        );

        let scalar2 = Scalar::struct_(
            dtype,
            vec![
                Scalar::primitive::<i32>(2, Nullability::NonNullable),
                Scalar::utf8("b", Nullability::NonNullable),
            ],
        );

        // Structs with same dtype can be compared
        assert!(scalar1.as_struct() < scalar2.as_struct());

        // Different struct types cannot be compared
        let other_dtype = DType::Struct(
            StructFields::new(
                ["c"].into(),
                vec![DType::Primitive(I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let scalar3 = Scalar::struct_(
            other_dtype,
            vec![Scalar::primitive::<i32>(1, Nullability::NonNullable)],
        );

        assert_eq!(scalar1.as_struct().partial_cmp(&scalar3.as_struct()), None);
    }

    #[test]
    fn test_struct_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let (_, _, dtype) = setup_types();

        let scalar1 = Scalar::struct_(
            dtype.clone(),
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        let scalar2 = Scalar::struct_(
            dtype,
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        let mut hasher1 = DefaultHasher::new();
        scalar1.as_struct().hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        scalar2.as_struct().hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_struct_try_new_non_struct_dtype() {
        let dtype = DType::Primitive(I32, Nullability::NonNullable);
        let value = ScalarValue(InnerScalarValue::Primitive(crate::PValue::I32(42)));

        let result = StructScalar::try_new(&dtype, &value);
        assert!(result.is_err());
    }

    #[test]
    fn test_struct_field_out_of_bounds() {
        let (_, _, dtype) = setup_types();
        let scalar = Scalar::struct_(
            dtype,
            vec![
                Scalar::primitive::<i32>(1, Nullability::NonNullable),
                Scalar::utf8("test", Nullability::NonNullable),
            ],
        );

        // Try to access field beyond bounds
        let field = scalar.as_struct().field_by_idx(10);
        assert!(field.is_none());
    }
}
