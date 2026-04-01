// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::iter::once;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::Struct;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::StructFields;
use crate::stats::ArrayStats;
use crate::validity::Validity;

// StructArray has a variable number of slots: [validity?, field_0, ..., field_N]
/// The validity bitmap indicating which struct elements are non-null.
pub(super) const VALIDITY_SLOT: usize = 0;
/// The offset at which the struct field arrays begin in the slots vector.
pub(super) const FIELDS_OFFSET: usize = 1;

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
/// use vortex_array::dtype::FieldNames;
/// use vortex_array::IntoArray;
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
/// use vortex_array::dtype::FieldNames;
/// use vortex_array::IntoArray;
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
/// use vortex_array::dtype::FieldNames;
/// use vortex_array::IntoArray;
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
pub struct StructData {
    pub(super) len: usize,
    pub(super) dtype: DType,
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) stats_set: ArrayStats,
}

pub struct StructArrayParts {
    pub struct_fields: StructFields,
    pub fields: Arc<[ArrayRef]>,
    pub validity: Validity,
}

impl StructData {
    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reconstructs the validity from the slots.
    pub fn validity(&self) -> Validity {
        child_to_validity(&self.slots[VALIDITY_SLOT], self.dtype.nullability())
    }

    /// Return an iterator over the struct fields without the validity of the struct applied.
    pub fn iter_unmasked_fields(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        self.slots[FIELDS_OFFSET..]
            .iter()
            .map(|s| s.as_ref().vortex_expect("StructArray field slot"))
    }

    /// Return the struct fields without the validity of the struct applied.
    pub fn unmasked_fields(&self) -> Arc<[ArrayRef]> {
        self.iter_unmasked_fields().cloned().collect()
    }

    /// Return the struct field at the given index without the validity of the struct applied.
    pub fn unmasked_field(&self, idx: usize) -> &ArrayRef {
        self.slots[FIELDS_OFFSET + idx]
            .as_ref()
            .vortex_expect("StructArray field slot")
    }

    /// Return the struct field without the validity of the struct applied
    pub fn unmasked_field_by_name(&self, name: impl AsRef<str>) -> VortexResult<&ArrayRef> {
        let name = name.as_ref();
        self.unmasked_field_by_name_opt(name).ok_or_else(|| {
            vortex_err!(
                "Field {name} not found in struct array with names {:?}",
                self.names()
            )
        })
    }

    /// Return the struct field without the validity of the struct applied
    pub fn unmasked_field_by_name_opt(&self, name: impl AsRef<str>) -> Option<&ArrayRef> {
        let name = name.as_ref();
        self.struct_fields().find(name).map(|idx| {
            self.slots[FIELDS_OFFSET + idx]
                .as_ref()
                .vortex_expect("StructArray field slot")
        })
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

    /// Creates a new `StructArray`.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented
    /// in `StructArray::new_unchecked`.
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
    /// See `StructArray::new_unchecked` for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// `StructArray::new_unchecked`.
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

    /// Creates a new `StructArray` without validation from these components:
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

        let validity_slot = validity_to_child(&validity, length);
        let slots = once(validity_slot)
            .chain(fields.iter().map(|f| Some(f.clone())))
            .collect();

        Self {
            len: length,
            dtype: DType::Struct(dtype, validity.nullability()),
            slots,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a `StructArray`.
    ///
    /// This function checks all the invariants required by `StructArray::new_unchecked`.
    pub fn validate(
        fields: &[ArrayRef],
        dtype: &StructFields,
        length: usize,
        validity: &Validity,
    ) -> VortexResult<()> {
        // Check field count matches
        if fields.len() != dtype.names().len() {
            vortex_bail!(
                InvalidArgument: "Got {} fields but dtype has {} names",
                fields.len(),
                dtype.names().len()
            );
        }

        // Check each field's length and dtype
        for (i, (field, struct_dt)) in fields.iter().zip(dtype.fields()).enumerate() {
            if field.len() != length {
                vortex_bail!(
                    InvalidArgument: "Field {} has length {} but expected {}",
                    i,
                    field.len(),
                    length
                );
            }

            if field.dtype() != &struct_dt {
                vortex_bail!(
                    InvalidArgument: "Field {} has dtype {} but expected {}",
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
                InvalidArgument: "Validity has length {} but expected {}",
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
        let validity = self.validity();
        let struct_fields = self.dtype.into_struct_fields();
        let fields: Arc<[ArrayRef]> = self
            .slots
            .into_iter()
            .skip(FIELDS_OFFSET)
            .map(|s| s.vortex_expect("StructArray field slot"))
            .collect();
        StructArrayParts {
            struct_fields,
            fields,
            validity,
        }
    }

    pub fn into_fields(self) -> Vec<ArrayRef> {
        self.into_parts().fields.to_vec()
    }

    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayRef)]) -> VortexResult<Self> {
        Self::try_from_iter(items.iter().map(|(a, b)| (a, b.clone())))
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

        for f_name in projection.iter() {
            let idx = self
                .names()
                .iter()
                .position(|name| name == f_name)
                .ok_or_else(|| vortex_err!("Unknown field {f_name}"))?;

            names.push(self.names()[idx].clone());
            children.push(
                self.slots[FIELDS_OFFSET + idx]
                    .as_ref()
                    .vortex_expect("StructArray field slot")
                    .clone(),
            );
        }

        StructData::try_new(
            FieldNames::from(names.as_slice()),
            children,
            self.len(),
            self.validity(),
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

        let slot_position = FIELDS_OFFSET + position;
        let field = self.slots[slot_position]
            .as_ref()
            .vortex_expect("StructArray field slot")
            .clone();
        let new_slots: Vec<Option<ArrayRef>> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != slot_position)
            .map(|(_, s)| s.clone())
            .collect();

        if let Ok(new_dtype) = struct_dtype.without_field(position) {
            self.slots = new_slots;
            self.dtype = DType::Struct(new_dtype, self.dtype.nullability());
            return Some(field);
        }
        None
    }
}

impl Array<Struct> {
    /// Creates a new `StructArray`.
    pub fn new(
        names: FieldNames,
        fields: impl Into<Arc<[ArrayRef]>>,
        length: usize,
        validity: Validity,
    ) -> Self {
        Array::try_from_data(StructData::new(names, fields, length, validity))
            .vortex_expect("StructData is always valid")
    }

    /// Constructs a new `StructArray`.
    pub fn try_new(
        names: FieldNames,
        fields: impl Into<Arc<[ArrayRef]>>,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        Array::try_from_data(StructData::try_new(names, fields, length, validity)?)
    }

    /// Creates a new `StructArray` without validation.
    ///
    /// # Safety
    ///
    /// See [`StructData::new_unchecked`].
    pub unsafe fn new_unchecked(
        fields: impl Into<Arc<[ArrayRef]>>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> Self {
        Array::try_from_data(unsafe { StructData::new_unchecked(fields, dtype, length, validity) })
            .vortex_expect("StructData is always valid")
    }

    /// Constructs a new `StructArray` with an explicit dtype.
    pub fn try_new_with_dtype(
        fields: impl Into<Arc<[ArrayRef]>>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        Array::try_from_data(StructData::try_new_with_dtype(
            fields, dtype, length, validity,
        )?)
    }

    /// Construct a `StructArray` from named fields.
    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayRef)]) -> VortexResult<Self> {
        Array::try_from_data(StructData::from_fields(items)?)
    }

    /// Decompose this struct array into its constituent parts.
    pub fn into_parts(self) -> StructArrayParts {
        self.into_data().into_parts()
    }

    /// Create a `StructArray` from an iterator of (name, array) pairs with validity.
    pub fn try_from_iter_with_validity<
        N: AsRef<str>,
        A: IntoArray,
        T: IntoIterator<Item = (N, A)>,
    >(
        iter: T,
        validity: Validity,
    ) -> VortexResult<Self> {
        Array::try_from_data(StructData::try_from_iter_with_validity(iter, validity)?)
    }

    /// Create a `StructArray` from an iterator of (name, array) pairs.
    pub fn try_from_iter<N: AsRef<str>, A: IntoArray, T: IntoIterator<Item = (N, A)>>(
        iter: T,
    ) -> VortexResult<Self> {
        Array::try_from_data(StructData::try_from_iter(iter)?)
    }

    /// Create a fieldless `StructArray` with the given length.
    pub fn new_fieldless_with_len(len: usize) -> Self {
        Array::try_from_data(StructData::new_fieldless_with_len(len))
            .vortex_expect("StructData is always valid")
    }
}

impl StructData {
    pub fn with_column(&self, name: impl Into<FieldName>, array: ArrayRef) -> VortexResult<Self> {
        let name = name.into();
        let struct_dtype = self.struct_fields().clone();

        let names = struct_dtype.names().iter().cloned().chain(once(name));
        let types = struct_dtype.fields().chain(once(array.dtype().clone()));
        let new_fields = StructFields::new(names.collect(), types.collect());

        let children: Arc<[ArrayRef]> = self.slots[FIELDS_OFFSET..]
            .iter()
            .map(|s| s.as_ref().vortex_expect("StructArray field slot").clone())
            .chain(once(array))
            .collect();

        Self::try_new_with_dtype(children, new_fields, self.len, self.validity())
    }
}
