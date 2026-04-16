// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::Struct;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::StructFields;
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
/// use vortex_array::{IntoArray, LEGACY_SESSION, VortexSessionExecute};
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
/// let mut ctx = LEGACY_SESSION.create_execution_ctx();
///
/// // Row 0 is valid - returns a struct scalar with field values
/// let row0 = struct_array.execute_scalar(0, &mut ctx).unwrap();
/// assert!(!row0.is_null());
///
/// // Row 1 is null at struct level - returns null even though fields have values
/// let row1 = struct_array.execute_scalar(1, &mut ctx).unwrap();
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
/// use vortex_array::arrays::struct_::StructArrayExt;
/// use vortex_array::validity::Validity;
/// use vortex_array::dtype::FieldNames;
/// use vortex_array::{IntoArray, LEGACY_SESSION, VortexSessionExecute};
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
/// let mut ctx = LEGACY_SESSION.create_execution_ctx();
/// assert_eq!(first_data.execute_scalar(0, &mut ctx).unwrap(), 1i32.into());
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
/// use vortex_array::arrays::struct_::StructArrayExt;
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
pub struct StructDataParts {
    pub struct_fields: StructFields,
    pub fields: Arc<[ArrayRef]>,
    pub validity: Validity,
}

pub(super) fn make_struct_slots(
    fields: &[ArrayRef],
    validity: &Validity,
    length: usize,
) -> Vec<Option<ArrayRef>> {
    once(validity_to_child(validity, length))
        .chain(fields.iter().cloned().map(Some))
        .collect()
}

pub trait StructArrayExt: TypedArrayRef<Struct> {
    fn nullability(&self) -> crate::dtype::Nullability {
        match self.as_ref().dtype() {
            DType::Struct(_, nullability) => *nullability,
            _ => unreachable!("StructArrayExt requires a struct dtype"),
        }
    }

    fn names(&self) -> &FieldNames {
        self.as_ref().dtype().as_struct_fields().names()
    }

    fn struct_validity(&self) -> Validity {
        child_to_validity(&self.as_ref().slots()[VALIDITY_SLOT], self.nullability())
    }

    fn iter_unmasked_fields(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        self.as_ref().slots()[FIELDS_OFFSET..]
            .iter()
            .map(|s| s.as_ref().vortex_expect("StructArray field slot"))
    }

    fn unmasked_fields(&self) -> Arc<[ArrayRef]> {
        self.iter_unmasked_fields().cloned().collect()
    }

    fn unmasked_field(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[FIELDS_OFFSET + idx]
            .as_ref()
            .vortex_expect("StructArray field slot")
    }

    fn unmasked_field_by_name_opt(&self, name: impl AsRef<str>) -> Option<&ArrayRef> {
        let name = name.as_ref();
        self.struct_fields()
            .find(name)
            .map(|idx| self.unmasked_field(idx))
    }

    fn unmasked_field_by_name(&self, name: impl AsRef<str>) -> VortexResult<&ArrayRef> {
        let name = name.as_ref();
        self.unmasked_field_by_name_opt(name).ok_or_else(|| {
            vortex_err!(
                "Field {name} not found in struct array with names {:?}",
                self.names()
            )
        })
    }

    fn struct_fields(&self) -> &StructFields {
        self.as_ref().dtype().as_struct_fields()
    }
}
impl<T: TypedArrayRef<Struct>> StructArrayExt for T {}

impl Array<Struct> {
    /// Creates a new `StructArray`.
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
    pub fn try_new(
        names: FieldNames,
        fields: impl Into<Arc<[ArrayRef]>>,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        let fields = fields.into();
        let field_dtypes: Vec<_> = fields.iter().map(|d| d.dtype().clone()).collect();
        let dtype = StructFields::new(names, field_dtypes);
        let slots = make_struct_slots(&fields, &validity, length);
        Array::try_from_parts(
            ArrayParts::new(
                Struct,
                DType::Struct(dtype, validity.nullability()),
                length,
                EmptyArrayData,
            )
            .with_slots(slots),
        )
    }

    /// Creates a new `StructArray` without validation.
    ///
    /// # Safety
    ///
    /// Caller must ensure the field arrays match the supplied dtype, length, and validity.
    pub unsafe fn new_unchecked(
        fields: impl Into<Arc<[ArrayRef]>>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> Self {
        let fields = fields.into();
        let outer_dtype = DType::Struct(dtype, validity.nullability());
        let slots = make_struct_slots(&fields, &validity, length);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Struct, outer_dtype, length, EmptyArrayData).with_slots(slots),
            )
        }
    }

    /// Constructs a new `StructArray` with an explicit dtype.
    pub fn try_new_with_dtype(
        fields: impl Into<Arc<[ArrayRef]>>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        let fields = fields.into();
        let outer_dtype = DType::Struct(dtype, validity.nullability());
        let slots = make_struct_slots(&fields, &validity, length);
        Array::try_from_parts(
            ArrayParts::new(Struct, outer_dtype, length, EmptyArrayData).with_slots(slots),
        )
    }

    /// Construct a `StructArray` from named fields.
    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayRef)]) -> VortexResult<Self> {
        Self::try_from_iter(items.iter().map(|(a, b)| (a, b.clone())))
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

    /// Create a `StructArray` from an iterator of (name, array) pairs.
    pub fn try_from_iter<N: AsRef<str>, A: IntoArray, T: IntoIterator<Item = (N, A)>>(
        iter: T,
    ) -> VortexResult<Self> {
        let (names, fields): (Vec<FieldName>, Vec<ArrayRef>) = iter
            .into_iter()
            .map(|(name, field)| (FieldName::from(name.as_ref()), field.into_array()))
            .unzip();
        let len = fields
            .first()
            .map(ArrayRef::len)
            .ok_or_else(|| vortex_err!("StructArray cannot be constructed from an empty slice of arrays because the length is unspecified"))?;

        Self::try_new(
            FieldNames::from_iter(names),
            fields,
            len,
            Validity::NonNullable,
        )
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

        for f_name in projection {
            let idx = self
                .struct_fields()
                .find(f_name.as_ref())
                .ok_or_else(|| vortex_err!("Unknown field {f_name}"))?;

            names.push(self.names()[idx].clone());
            children.push(self.unmasked_field(idx).clone());
        }

        Self::try_new(
            FieldNames::from(names.as_slice()),
            children,
            self.len(),
            self.validity()?,
        )
    }

    /// Create a fieldless `StructArray` with the given length.
    pub fn new_fieldless_with_len(len: usize) -> Self {
        let dtype = DType::Struct(
            StructFields::new(FieldNames::default(), Vec::new()),
            crate::dtype::Nullability::NonNullable,
        );
        let slots = make_struct_slots(&[], &Validity::NonNullable, len);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Struct, dtype, len, EmptyArrayData).with_slots(slots),
            )
        }
    }

    // TODO(ngates): remove this... it doesn't help to consume self.
    pub fn into_data_parts(self) -> StructDataParts {
        let fields: Arc<[ArrayRef]> = self.slots()[FIELDS_OFFSET..]
            .iter()
            .map(|s| s.as_ref().vortex_expect("StructArray field slot").clone())
            .collect();
        let validity = self.validity().vortex_expect("StructArray validity");
        StructDataParts {
            struct_fields: self.struct_fields().clone(),
            fields,
            validity,
        }
    }

    pub fn remove_column(&self, name: impl Into<FieldName>) -> Option<(Self, ArrayRef)> {
        let name = name.into();
        let struct_dtype = self.struct_fields();
        let len = self.len();

        let position = struct_dtype.find(name.as_ref())?;

        let slot_position = FIELDS_OFFSET + position;
        let field = self.slots()[slot_position]
            .as_ref()
            .vortex_expect("StructArray field slot")
            .clone();
        let new_slots: Vec<Option<ArrayRef>> = self
            .slots()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != slot_position)
            .map(|(_, s)| s.clone())
            .collect();

        let new_dtype = struct_dtype.without_field(position).ok()?;
        let new_array = unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(
                    Struct,
                    DType::Struct(new_dtype, self.dtype().nullability()),
                    len,
                    EmptyArrayData,
                )
                .with_slots(new_slots),
            )
        };
        Some((new_array, field))
    }
}

impl Array<Struct> {
    pub fn with_column(&self, name: impl Into<FieldName>, array: ArrayRef) -> VortexResult<Self> {
        let name = name.into();
        let struct_dtype = self.struct_fields();

        let names = struct_dtype.names().iter().cloned().chain(once(name));
        let types = struct_dtype.fields().chain(once(array.dtype().clone()));
        let new_fields = StructFields::new(names.collect(), types.collect());

        let children: Arc<[ArrayRef]> = self.slots()[FIELDS_OFFSET..]
            .iter()
            .map(|s| s.as_ref().vortex_expect("StructArray field slot").clone())
            .chain(once(array))
            .collect();

        Self::try_new_with_dtype(children, new_fields, self.len(), self.validity()?)
    }

    pub fn remove_column_owned(&self, name: impl Into<FieldName>) -> Option<(Self, ArrayRef)> {
        self.remove_column(name)
    }
}
