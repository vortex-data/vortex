// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use itertools::Itertools;
use vortex_dtype::{DType, FieldName, FieldNames, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use crate::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};

mod compute;
mod serde;

vtable!(Struct);

impl VTable for StructVTable {
    type Array = StructArray;
    type Encoding = StructEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.struct")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(StructEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct StructArray {
    len: usize,
    dtype: DType,
    fields: Vec<ArrayRef>,
    validity: Validity,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct StructEncoding;

impl StructArray {
    pub fn fields(&self) -> &[ArrayRef] {
        &self.fields
    }

    pub fn field_by_name(&self, name: impl AsRef<str>) -> VortexResult<&ArrayRef> {
        let name = name.as_ref();
        self.field_by_name_opt(name).ok_or_else(|| {
            vortex_err!(
                "Field {name} not found in struct array with names {:?}",
                self.names()
            )
        })
    }

    pub fn field_by_name_opt(&self, name: impl AsRef<str>) -> Option<&ArrayRef> {
        let name = name.as_ref();
        self.names()
            .iter()
            .position(|field_name| field_name.as_ref() == name)
            .map(|idx| &self.fields[idx])
    }

    pub fn names(&self) -> &FieldNames {
        self.struct_fields().names()
    }

    pub fn struct_fields(&self) -> &StructFields {
        let Some(struct_dtype) = &self.dtype.as_struct() else {
            unreachable!(
                "struct arrays must have be a DType::Struct, this is likely an internal bug."
            )
        };
        struct_dtype
    }

    pub fn try_new(
        names: FieldNames,
        fields: Vec<ArrayRef>,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        let nullability = validity.nullability();

        if names.len() != fields.len() {
            vortex_bail!("Got {} names and {} fields", names.len(), fields.len());
        }

        for field in fields.iter() {
            if field.len() != length {
                vortex_bail!(
                    "Expected all struct fields to have length {length}, found {}",
                    fields.iter().map(|f| f.len()).format(","),
                );
            }
        }

        let field_dtypes: Vec<_> = fields.iter().map(|d| d.dtype()).cloned().collect();
        let dtype = DType::Struct(StructFields::new(names, field_dtypes), nullability);

        if length != validity.maybe_len().unwrap_or(length) {
            vortex_bail!(
                "array length {} and validity length must match {}",
                length,
                validity
                    .maybe_len()
                    .vortex_expect("can only fail if maybe is some")
            )
        }

        Ok(Self {
            len: length,
            dtype,
            fields,
            validity,
            stats_set: Default::default(),
        })
    }

    pub fn try_new_with_dtype(
        fields: Vec<ArrayRef>,
        dtype: StructFields,
        length: usize,
        validity: Validity,
    ) -> VortexResult<Self> {
        for (field, struct_dt) in fields.iter().zip(dtype.fields()) {
            if field.len() != length {
                vortex_bail!(
                    "Expected all struct fields to have length {length}, found {}",
                    field.len()
                );
            }

            if &struct_dt != field.dtype() {
                vortex_bail!(
                    "Expected all struct fields to have dtype {}, found {}",
                    struct_dt,
                    field.dtype()
                );
            }
        }

        Ok(Self {
            len: length,
            dtype: DType::Struct(dtype, validity.nullability()),
            fields,
            validity,
            stats_set: Default::default(),
        })
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
    #[allow(clippy::same_name_method)]
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
            children.push(self.fields()[idx].clone());
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

        let Some(struct_dtype) = self.dtype.as_struct() else {
            unreachable!(
                "struct arrays must have be a DType::Struct, this is likely an internal bug."
            )
        };

        let position = struct_dtype
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name.as_ref())?;

        let field = self.fields.remove(position);

        let new_dtype = struct_dtype.without_field(position);
        self.dtype = DType::Struct(new_dtype, self.dtype.nullability());

        Some(field)
    }
}

impl ValidityHelper for StructArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<StructVTable> for StructVTable {
    fn len(array: &StructArray) -> usize {
        array.len
    }

    fn dtype(array: &StructArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &StructArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<StructVTable> for StructVTable {
    fn canonicalize(array: &StructArray) -> VortexResult<Canonical> {
        Ok(Canonical::Struct(array.clone()))
    }
}

impl OperationsVTable<StructVTable> for StructVTable {
    fn slice(array: &StructArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let fields = array
            .fields()
            .iter()
            .map(|field| field.slice(start, stop))
            .try_collect()?;
        StructArray::try_new_with_dtype(
            fields,
            array.struct_fields().clone(),
            stop - start,
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }

    fn scalar_at(array: &StructArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            array.dtype().clone(),
            array
                .fields()
                .iter()
                .map(|field| field.scalar_at(index))
                .try_collect()?,
        ))
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability, PType};

    use crate::IntoArray;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::arrays::struct_::StructArray;
    use crate::arrays::varbin::VarBinArray;
    use crate::arrays::{BoolArray, BoolVTable, PrimitiveVTable};
    use crate::validity::Validity;

    #[test]
    fn test_project() {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::NonNullable),
        );
        let zs = BoolArray::from_iter([true, true, true, false, false]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs", "ys", "zs"]),
            vec![xs.into_array(), ys.into_array(), zs.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap();

        let struct_b = struct_a
            .project(&[FieldName::from("zs"), FieldName::from("xs")])
            .unwrap();
        assert_eq!(
            struct_b.names().as_ref(),
            [FieldName::from("zs"), FieldName::from("xs")],
        );

        assert_eq!(struct_b.len(), 5);

        let bools = &struct_b.fields[0];
        assert_eq!(
            bools
                .as_::<BoolVTable>()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, true, true, false, false]
        );

        let prims = &struct_b.fields[1];
        assert_eq!(
            prims.as_::<PrimitiveVTable>().as_slice::<i64>(),
            [0i64, 1, 2, 3, 4]
        );
    }

    #[test]
    fn test_remove_column() {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
        let ys = PrimitiveArray::new(buffer![4u64, 5, 6, 7, 8], Validity::NonNullable);

        let mut struct_a = StructArray::try_new(
            FieldNames::from(["xs", "ys"]),
            vec![xs.into_array(), ys.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap();

        let removed = struct_a.remove_column("xs").unwrap();
        assert_eq!(
            removed.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_eq!(
            removed.as_::<PrimitiveVTable>().as_slice::<i64>(),
            [0i64, 1, 2, 3, 4]
        );

        assert_eq!(struct_a.names(), &[FieldName::from("ys")].into());
        assert_eq!(struct_a.fields.len(), 1);
        assert_eq!(struct_a.len(), 5);
        assert_eq!(
            struct_a.fields[0].dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable)
        );
        assert_eq!(
            struct_a.fields[0]
                .as_::<PrimitiveVTable>()
                .as_slice::<u64>(),
            [4u64, 5, 6, 7, 8]
        );

        let empty = struct_a.remove_column("non_existent");
        assert!(
            empty.is_none(),
            "Expected None when removing non-existent column"
        );
        assert_eq!(struct_a.names(), &[FieldName::from("ys")].into());
    }
}
