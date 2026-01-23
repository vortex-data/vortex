// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::iter::once;
use std::ops::Not;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldNames;
use vortex_dtype::Nullability;
use vortex_dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::builtins::ArrayBuiltins;
use crate::compute::mask;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

/// Metadata for StructArray serialization.
#[derive(Clone, prost::Message)]
pub struct StructMetadata {
    /// If true, child validity is a superset of struct validity (validity was pushed down).
    /// For nullable children, their validity already includes struct nulls. For non-nullable
    /// children, we apply struct validity on field read. If false (default), no guarantee
    /// about relationship - must intersect validities on read.
    #[prost(bool, tag = "1", default = false)]
    pub(super) validity_pushed_down: bool,
}

/// A struct array that stores multiple named fields as columns, similar to a database row.
///
/// This mirrors the Apache Arrow Struct array encoding and provides a columnar representation
/// of structured data where each row contains multiple named fields of potentially different types.
///
/// ## Data Layout
///
/// The struct array uses a columnar layout where:
/// - Each field is stored as a separate child array
/// - All fields must have the same length (number of rows)
/// - Field names and types are defined in the struct's dtype
/// - An optional validity mask indicates which entire rows are null
///
/// ## Row-level nulls
///
/// The StructArray contains its own top-level nulls, which are superimposed on top of the
/// field-level validity values. This can be the case even if the fields themselves are non-nullable,
/// accessing a particular row can yield nulls even if all children are valid at that position.
///
/// ```
/// use vortex_array::arrays::{StructArray, BoolArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_dtype::FieldNames;
/// use vortex_buffer::buffer;
///
/// // Create struct with all non-null fields but struct-level nulls
/// let struct_array = StructArray::try_new(
///     FieldNames::from(["a", "b", "c"]),
///     vec![
///         buffer![1i32, 2i32].into_array(),  // non-null field a
///         buffer![10i32, 20i32].into_array(), // non-null field b
///         buffer![100i32, 200i32].into_array(), // non-null field c
///     ],
///     2,
///     Validity::Array(BoolArray::from_iter([true, false]).into_array()), // row 1 is null
/// ).unwrap();
///
/// // Row 0 is valid - returns a struct scalar with field values
/// let row0 = struct_array.scalar_at(0).unwrap();
/// assert!(!row0.is_null());
///
/// // Row 1 is null at struct level - returns null even though fields have values
/// let row1 = struct_array.scalar_at(1).unwrap();
/// assert!(row1.is_null());
/// ```
///
/// ## Name uniqueness
///
/// It is valid for a StructArray to have multiple child columns that have the same name. In this
/// case, any accessors that use column names will find the first column in sequence with the name.
///
/// ```
/// use vortex_array::arrays::StructArray;
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_dtype::FieldNames;
/// use vortex_buffer::buffer;
///
/// // Create struct with duplicate "data" field names
/// let struct_array = StructArray::try_new(
///     FieldNames::from(["data", "data"]),
///     vec![
///         buffer![1i32, 2i32].into_array(),   // first "data"
///         buffer![3i32, 4i32].into_array(),   // second "data"
///     ],
///     2,
///     Validity::NonNullable,
/// ).unwrap();
///
/// // field_by_name returns the FIRST "data" field
/// let first_data = struct_array.unmasked_field_by_name("data").unwrap();
/// assert_eq!(first_data.scalar_at(0).unwrap(), 1i32.into());
/// ```
///
/// ## Field Operations
///
/// Struct arrays support efficient column operations:
/// - **Projection**: Select/reorder fields without copying data
/// - **Field access**: Get columns by name or index
/// - **Column addition**: Add new fields to create extended structs
/// - **Column removal**: Remove fields to create narrower structs
///
/// ## Validity Semantics
///
/// - Row-level nulls are tracked in the struct's validity child
/// - Individual field nulls are tracked in each field's own validity
/// - A null struct row means all fields in that row are conceptually null
/// - Field-level nulls can exist independently of struct-level nulls
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::{StructArray, PrimitiveArray};
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_dtype::FieldNames;
/// use vortex_buffer::buffer;
///
/// // Create arrays for each field
/// let ids = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
/// let names = PrimitiveArray::new(buffer![100u64, 200, 300], Validity::NonNullable);
///
/// // Create struct array with named fields
/// let struct_array = StructArray::try_new(
///     FieldNames::from(["id", "score"]),
///     vec![ids.into_array(), names.into_array()],
///     3,
///     Validity::NonNullable,
/// ).unwrap();
///
/// assert_eq!(struct_array.len(), 3);
/// assert_eq!(struct_array.names().len(), 2);
///
/// // Access field by name
/// let id_field = struct_array.unmasked_field_by_name("id").unwrap();
/// assert_eq!(id_field.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct StructArray {
    pub(super) len: usize,
    pub(super) dtype: DType,
    pub(super) fields: Arc<[ArrayRef]>,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
    /// true = child validity is a superset of struct validity (validity was pushed down)
    /// false = default, no guarantee about relationship
    pub(super) validity_pushed_down: bool,
}

pub struct StructArrayParts {
    pub struct_fields: StructFields,
    pub nullability: Nullability,
    pub fields: Arc<[ArrayRef]>,
    pub validity: Validity,
}

impl StructArray {
    /// Note this field may not have the validity of the parent struct applied.
    /// Should use `masked_fields` instead, unless you know what you are doing.
    pub fn unmasked_fields(&self) -> &Arc<[ArrayRef]> {
        &self.fields
    }

    pub fn masked_fields(&self) -> VortexResult<Vec<ArrayRef>> {
        if !self.dtype.is_nullable() {
            // fields need not be masked
            return Ok(self.fields.to_vec());
        }

        if self.has_validity_pushed_down() {
            self.fields
                .iter()
                .cloned()
                .map(|f| {
                    if f.dtype().is_nullable() {
                        Ok(f.into_array())
                    } else {
                        let validity = self.validity().to_array(self.len);
                        f.mask(validity)
                    }
                })
                .collect::<VortexResult<Vec<_>>>()
        } else {
            // Apply struct validity to all fields
            let struct_validity = self.validity().to_array(self.len);
            self.fields
                .iter()
                .map(move |f| f.clone().mask(struct_validity.clone()))
                .collect::<VortexResult<Vec<_>>>()
        }
    }

    /// Return the struct field with name `name` with the struct validity already applied.
    /// If the struct has no field with that `name` an error is returned.
    pub fn field_by_name(&self, name: impl AsRef<str>) -> VortexResult<ArrayRef> {
        let name = name.as_ref();
        self.field_by_name_opt(name)?.ok_or_else(|| {
            vortex_err!(
                "Field {name} not found in struct array with names {:?}",
                self.names()
            )
        })
    }

    /// Return the struct field with name `name` with the struct validity already applied.
    /// If the struct has no field with that `name` Ok(None) is returned.
    pub fn field_by_name_opt(&self, name: impl AsRef<str>) -> VortexResult<Option<ArrayRef>> {
        let name = name.as_ref();
        self.struct_fields()
            .find(name)
            .map(|idx| {
                let field = self.fields[idx].clone();
                // Non-nullable struct: return field as-is (no struct validity to apply)
                if !self.dtype.is_nullable() {
                    return Ok(field);
                }
                // Non-nullable field: always apply struct validity (even with validity_pushed_down,
                // since we can't push validity to non-nullable fields)
                if !field.dtype().is_nullable() {
                    return field.mask(self.validity().to_array(self.len));
                }
                // Nullable field: return as-is (validity is either in the field or was pushed down)
                Ok(field)
            })
            .transpose()
    }

    /// Note this field may not have the validity of the parent struct applied.
    /// Should use `field_by_name` instead.
    pub fn unmasked_field_by_name(&self, name: impl AsRef<str>) -> VortexResult<&ArrayRef> {
        let name = name.as_ref();
        self.unmasked_field_by_name_opt(name).ok_or_else(|| {
            vortex_err!(
                "Field {name} not found in struct array with names {:?}",
                self.names()
            )
        })
    }

    /// Note this field may not have the validity of the parent struct applied.
    /// Should use `field_by_name_opt` instead.
    pub fn unmasked_field_by_name_opt(&self, name: impl AsRef<str>) -> Option<&ArrayRef> {
        let name = name.as_ref();
        self.struct_fields().find(name).map(|idx| &self.fields[idx])
    }

    pub fn names(&self) -> &FieldNames {
        self.struct_fields().names()
    }

    pub fn struct_fields(&self) -> &StructFields {
        let Some(struct_dtype) = &self.dtype.as_struct_fields_opt() else {
            unreachable!(
                "struct arrays must have be a DType::Struct, this is likely an internal bug."
            )
        };
        struct_dtype
    }

    /// Create a new `StructArray` with the given length, but without any fields.
    pub fn new_fieldless_with_len(len: usize) -> Self {
        Self::try_new(
            FieldNames::default(),
            Vec::new(),
            len,
            Validity::NonNullable,
        )
        .vortex_expect("StructArray::new_with_len should not fail")
    }

    /// Creates a new [`StructArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in [`StructArray::new_unchecked`].
    pub fn new(
        names: FieldNames,
        fields: impl Into<Arc<[ArrayRef]>>,
        length: usize,
        validity: Validity,
    ) -> Self {
        Self::try_new(names, fields, length, validity)
            .vortex_expect("StructArray construction failed")
    }

    /// Constructs a new `StructArray`.
    ///
    /// See [`StructArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`StructArray::new_unchecked`].
    pub fn try_new(
        names: FieldNames,
        fields: impl Into<Arc<[ArrayRef]>>,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        let fields = fields.into();
        let field_dtypes: Vec<_> = fields.iter().map(|d| d.dtype()).cloned().collect();
        let dtype = StructFields::new(names, field_dtypes);

        Self::validate(&fields, &dtype, length, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(fields, dtype, length, validity) })
    }

    /// Creates a new [`StructArray`] without validation from these components:
    ///
    /// * `fields` is a vector of arrays, one for each field in the struct.
    /// * `dtype` contains the field names and types.
    /// * `length` is the number of struct rows.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// ## Field Requirements
    ///
    /// - `fields.len()` must exactly equal `dtype.names().len()`.
    /// - Every field array in `fields` must have length exactly equal to `length`.
    /// - For each index `i`, `fields[i].dtype()` must exactly match `dtype.fields()[i]`.
    ///
    /// ## Type Requirements
    ///
    /// - Field names in `dtype` may be duplicated (this is explicitly allowed).
    /// - The nullability of `dtype` must match the nullability of `validity`.
    ///
    /// ## Validity Requirements
    ///
    /// - If `validity` is [`Validity::Array`], its length must exactly equal `length`.
    pub unsafe fn new_unchecked(
        fields: impl Into<Arc<[ArrayRef]>>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> Self {
        let fields = fields.into();

        #[cfg(debug_assertions)]
        Self::validate(&fields, &dtype, length, &validity)
            .vortex_expect("[Debug Assertion]: Invalid `StructArray` parameters");

        Self {
            len: length,
            dtype: DType::Struct(dtype, validity.nullability()),
            fields,
            validity,
            stats_set: Default::default(),
            validity_pushed_down: false,
        }
    }

    /// Validates the components that would be used to create a [`StructArray`].
    ///
    /// This function checks all the invariants required by [`StructArray::new_unchecked`].
    pub fn validate(
        fields: &[ArrayRef],
        dtype: &StructFields,
        length: usize,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Check field count matches
        if fields.len() != dtype.names().len() {
            vortex_bail!(
                "Got {} fields but dtype has {} names",
                fields.len(),
                dtype.names().len()
            );
        }

        // Check each field's length and dtype
        for (i, (field, struct_dt)) in fields.iter().zip(dtype.fields()).enumerate() {
            if field.len() != length {
                vortex_bail!(
                    "Field {} has length {} but expected {}",
                    i,
                    field.len(),
                    length
                );
            }

            if field.dtype() != &struct_dt {
                vortex_bail!(
                    "Field {} has dtype {} but expected {}",
                    i,
                    field.dtype(),
                    struct_dt
                );
            }
        }

        // Check validity length
        if let Some(validity_len) = validity.maybe_len()
            && validity_len != length
        {
            vortex_bail!(
                "Validity has length {} but expected {}",
                validity_len,
                length
            );
        }

        Ok(())
    }

    pub fn try_new_with_dtype(
        fields: impl Into<Arc<[ArrayRef]>>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        let fields = fields.into();
        Self::validate(&fields, &dtype, length, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(fields, dtype, length, validity) })
    }

    pub fn into_parts(self) -> StructArrayParts {
        let nullability = self.dtype.nullability();
        let struct_fields = self.dtype.into_struct_fields();
        StructArrayParts {
            struct_fields,
            nullability,
            fields: self.fields,
            validity: self.validity,
        }
    }

    pub fn into_fields(self) -> Vec<ArrayRef> {
        self.into_parts().fields.to_vec()
    }

    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayRef)]) -> VortexResult<Self> {
        Self::try_from_iter(items.iter().map(|(a, b)| (a, b.to_array())))
    }

    pub fn try_from_iter_with_validity<
        N: AsRef<str>,
        A: IntoArray,
        T: IntoIterator<Item = (N, A)>,
    >(
        iter: T,
        validity: Validity,
    ) -> VortexResult<Self> {
        let (names, fields): (Vec<FieldName>, Vec<ArrayRef>) = iter
            .into_iter()
            .map(|(name, fields)| (FieldName::from(name.as_ref()), fields.into_array()))
            .unzip();
        let len = fields
            .first()
            .map(|f| f.len())
            .ok_or_else(|| vortex_err!("StructArray cannot be constructed from an empty slice of arrays because the length is unspecified"))?;

        Self::try_new(FieldNames::from_iter(names), fields, len, validity)
    }

    pub fn try_from_iter<N: AsRef<str>, A: IntoArray, T: IntoIterator<Item = (N, A)>>(
        iter: T,
    ) -> VortexResult<Self> {
        Self::try_from_iter_with_validity(iter, Validity::NonNullable)
    }

    // TODO(aduffy): Add equivalent function to support field masks for nested column access.
    /// Return a new StructArray with the given projection applied.
    ///
    /// Projection does not copy data arrays. Projection is defined by an ordinal array slice
    /// which specifies the new ordering of columns in the struct. The projection can be used to
    /// perform column re-ordering, deletion, or duplication at a logical level, without any data
    /// copying.
    pub fn project(&self, projection: &[FieldName]) -> VortexResult<Self> {
        let mut children = Vec::with_capacity(projection.len());
        let mut names = Vec::with_capacity(projection.len());

        let fields = self.unmasked_fields();
        for f_name in projection.iter() {
            let idx = self
                .names()
                .iter()
                .position(|name| name == f_name)
                .ok_or_else(|| vortex_err!("Unknown field {f_name}"))?;

            names.push(self.names()[idx].clone());
            children.push(fields[idx].clone());
        }

        StructArray::try_new(
            FieldNames::from(names.as_slice()),
            children,
            self.len(),
            self.validity().clone(),
        )
    }

    /// Removes and returns a column from the struct array by name.
    /// If the column does not exist, returns `None`.
    pub fn remove_column(&mut self, name: impl Into<FieldName>) -> Option<ArrayRef> {
        let name = name.into();

        let struct_dtype = self.struct_fields().clone();

        let position = struct_dtype
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name.as_ref())?;

        let field = self.fields[position].clone();
        let new_fields: Arc<[ArrayRef]> = self
            .fields
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != position)
            .map(|(_, f)| f.clone())
            .collect();

        if let Ok(new_dtype) = struct_dtype.without_field(position) {
            self.fields = new_fields;
            self.dtype = DType::Struct(new_dtype, self.dtype.nullability());
            return Some(field);
        }
        None
    }

    /// Create a new StructArray by appending a new column onto the existing array.
    pub fn with_column(&self, name: impl Into<FieldName>, array: ArrayRef) -> VortexResult<Self> {
        let name = name.into();
        let struct_dtype = self.struct_fields().clone();

        let names = struct_dtype.names().iter().cloned().chain(once(name));
        let types = struct_dtype.fields().chain(once(array.dtype().clone()));
        let new_fields = StructFields::new(names.collect(), types.collect());

        let children: Arc<[ArrayRef]> = self.fields.iter().cloned().chain(once(array)).collect();

        Self::try_new_with_dtype(children, new_fields, self.len, self.validity.clone())
    }

    /// Returns whether validity has been pushed down into children.
    ///
    /// When true, child validity is a superset of struct validity (children include
    /// the struct's nulls baked in). This is an optimization that allows readers to
    /// skip combining struct+child validity when extracting fields.
    pub fn has_validity_pushed_down(&self) -> bool {
        #[cfg(debug_assertions)]
        if self.validity_pushed_down {
            self.validate_validity_pushed_down()
                .vortex_expect("validity_pushed_down invariant violated");
        }
        self.validity_pushed_down
    }

    /// Checks that the validity_pushed_down invariant holds.
    ///
    /// When `validity_pushed_down` is true, for every nullable child field,
    /// the child's validity must be a superset of the struct's validity.
    /// That is, wherever the struct is invalid (null), the child must also be invalid.
    fn validate_validity_pushed_down(&self) -> VortexResult<()> {
        if !self.validity_pushed_down {
            return Ok(());
        }

        let struct_validity = self.validity_mask()?;

        for (idx, field) in self.fields.iter().enumerate() {
            // Only check nullable children - non-nullable children cannot have validity pushed down
            if !field.dtype().is_nullable() {
                continue;
            }

            let child_validity = field.validity_mask()?;

            // Check invariant: struct_invalid => child_invalid
            // Equivalently: (!struct_validity) & child_validity should be all-false
            // If struct is invalid (false) but child is valid (true), that's a violation
            let violation = &(!&struct_validity) & &child_validity;
            if !violation.all_false() {
                vortex_bail!(
                    "validity_pushed_down invariant violated for field {}: \
                     struct has nulls at positions where child is valid",
                    idx
                );
            }
        }

        Ok(())
    }

    /// Set the validity_pushed_down flag.
    ///
    /// For non-nullable structs, this is a no-op (flag stays false) since there's no validity to
    /// push down
    ///
    /// For nullable structs, setting this to true indicates that child validity
    /// is a superset of struct validity (children include struct's nulls).
    ///
    /// # Safety
    ///
    /// If set all non-nullable field must have their nullability be a superset of the struct
    /// validity
    pub unsafe fn with_validity_pushed_down(mut self, validity_pushed_down: bool) -> Self {
        // For non-nullable structs, the flag is meaningless - keep it false
        if !self.dtype.is_nullable() {
            return self;
        }
        self.validity_pushed_down = validity_pushed_down;

        #[cfg(debug_assertions)]
        if validity_pushed_down {
            self.validate_validity_pushed_down()
                .vortex_expect("validity_pushed_down invariant violated");
        }

        self
    }

    /// Push struct validity down into each child field.
    ///
    /// For nullable structs with non-trivial validity, this applies the validity
    /// mask to each **nullable** child field, making child validity a superset
    /// of parent validity. Non-nullable children are left unchanged to preserve
    /// their dtype.
    ///
    /// The struct validity is **preserved** (DType never changes). The
    /// `validity_pushed_down` flag indicates that nullable children already include
    /// the parent's nulls, so readers can skip combining validities for those fields.
    ///
    /// For non-nullable structs or trivial validity, this is essentially a no-op.
    pub fn compact(&self) -> VortexResult<Self> {
        // For non-nullable structs, nothing to push down
        if !self.dtype.is_nullable() {
            return Ok(self.clone());
        }

        // If validity is trivial (AllValid), nothing to push down
        // but mark as pushed down since children trivially include parent validity
        if self.validity.all_valid(self.len)? {
            // # Safety no validity to push down
            return Ok(unsafe { self.clone().with_validity_pushed_down(true) });
        }

        // Get the validity mask - mask() expects true = set to null, so we invert the validity
        let validity_mask = self.validity_mask()?.not();

        // Apply mask only to nullable children - non-nullable children cannot have their
        // dtype changed, so we leave them alone
        let new_fields: Vec<ArrayRef> = self
            .unmasked_fields()
            .iter()
            .map(|field| {
                if field.dtype().is_nullable() {
                    mask(field.as_ref(), &validity_mask)
                } else {
                    Ok(field.clone())
                }
            })
            .collect::<VortexResult<_>>()?;

        // Create new struct with same validity but updated children
        // # Safety mask of struct validity applied to each child.
        Ok(unsafe {
            StructArray::try_new(
                self.names().clone(),
                new_fields,
                self.len(),
                self.validity.clone(), // Keep original validity
            )?
            .with_validity_pushed_down(true)
        })
    }
}
