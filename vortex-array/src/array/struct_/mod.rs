use std::fmt::{Debug, Display};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, Field, FieldName, FieldNames, StructDType};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::arrow::IntoArrowArray;
use crate::encoding::ids;
use crate::stats::{ArrayStatistics, Stat, StatsSet};
use crate::validity::{Validity, ValidityMetadata};
use crate::variants::StructArrayTrait;
use crate::visitor::ArrayVisitor;
use crate::vtable::{
    StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable, VisitorVTable,
};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, Canonical, DeserializeMetadata, IntoArrayData,
    IntoCanonical, RkyvMetadata,
};

mod compute;

impl_encoding!(
    "vortex.struct",
    ids::STRUCT,
    Struct,
    RkyvMetadata<StructMetadata>
);

#[derive(
    Clone, Debug, Serialize, Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[repr(C)]
pub struct StructMetadata {
    pub(crate) validity: ValidityMetadata,
}

impl Display for StructMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl StructArray {
    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(self.nfields(), &Validity::DTYPE, self.len())
                .vortex_expect("StructArray: validity child")
        })
    }

    pub fn children(&self) -> impl Iterator<Item = ArrayData> + '_ {
        (0..self.nfields()).map(move |idx| {
            self.maybe_null_field_by_idx(idx).unwrap_or_else(|| {
                vortex_panic!("Field {} not found, nfields: {}", idx, self.nfields())
            })
        })
    }

    pub fn try_new(
        names: FieldNames,
        fields: Vec<ArrayData>,
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
                    field.len()
                );
            }
        }

        let field_dtypes: Vec<_> = fields.iter().map(|d| d.dtype()).cloned().collect();

        let validity_metadata = validity.to_metadata(length)?;

        let mut children = Vec::with_capacity(fields.len() + 1);
        children.extend(fields);
        if let Some(v) = validity.into_array() {
            children.push(v);
        }

        Self::try_from_parts(
            DType::Struct(Arc::new(StructDType::new(names, field_dtypes)), nullability),
            length,
            RkyvMetadata(StructMetadata {
                validity: validity_metadata,
            }),
            None,
            Some(children.into()),
            StatsSet::default(),
        )
    }

    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayData)]) -> VortexResult<Self> {
        let names = items.iter().map(|(name, _)| FieldName::from(name.as_ref()));
        let fields: Vec<ArrayData> = items.iter().map(|(_, array)| array.clone()).collect();
        let len = fields
            .first()
            .map(|f| f.len())
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
            children.push(
                self.maybe_null_field_by_idx(idx)
                    .ok_or_else(|| vortex_err!(OutOfBounds: idx, 0, self.nfields()))?,
            );
        }

        StructArray::try_new(
            FieldNames::from(names.as_slice()),
            children,
            self.len(),
            self.validity(),
        )
    }
}

impl ValidateVTable<StructArray> for StructEncoding {}

impl VariantsVTable<StructArray> for StructEncoding {
    fn as_struct_array<'a>(&self, array: &'a StructArray) -> Option<&'a dyn StructArrayTrait> {
        Some(array)
    }
}

impl StructArrayTrait for StructArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> Option<ArrayData> {
        Some(
            self.field_info(&Field::Index(idx))
                .map(|field_info| {
                    self.as_ref()
                        .child(
                            idx,
                            &field_info
                                .dtype
                                .value()
                                .vortex_expect("FieldInfo could not access dtype"),
                            self.len(),
                        )
                        .unwrap_or_else(|e| {
                            vortex_panic!(e, "StructArray: field {} not found", idx)
                        })
                })
                .unwrap_or_else(|e| vortex_panic!(e, "StructArray: field {} not found", idx)),
        )
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayData> {
        self.project(projection).map(|a| a.into_array())
    }
}

impl IntoCanonical for StructArray {
    /// StructEncoding is the canonical form for a [DType::Struct] array, so return self.
    fn into_canonical(self) -> VortexResult<Canonical> {
        Ok(Canonical::Struct(self))
    }
}

impl ValidityVTable<StructArray> for StructEncoding {
    fn is_valid(&self, array: &StructArray, index: usize) -> VortexResult<bool> {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &StructArray) -> VortexResult<Mask> {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<StructArray> for StructEncoding {
    fn accept(&self, array: &StructArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        for (idx, name) in array.names().iter().enumerate() {
            let child = array
                .maybe_null_field_by_idx(idx)
                .ok_or_else(|| vortex_err!(OutOfBounds: idx, 0, array.nfields()))?;
            visitor.visit_child(name.as_ref(), &child)?;
        }
        Ok(())
    }
}

impl StatisticsVTable<StructArray> for StructEncoding {
    fn compute_statistics(&self, array: &StructArray, stat: Stat) -> VortexResult<StatsSet> {
        Ok(match stat {
            Stat::UncompressedSizeInBytes => array
                .children()
                .map(|f| f.statistics().compute_uncompressed_size_in_bytes())
                .reduce(|acc, field_size| acc.zip(field_size).map(|(a, b)| a + b))
                .flatten()
                .map(|size| StatsSet::of(stat, size))
                .unwrap_or_default(),
            Stat::NullCount => StatsSet::of(stat, array.validity().null_count(array.len())?),
            _ => StatsSet::default(),
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability};

    use crate::array::primitive::PrimitiveArray;
    use crate::array::struct_::StructArray;
    use crate::array::varbin::VarBinArray;
    use crate::array::BoolArray;
    use crate::validity::Validity;
    use crate::variants::StructArrayTrait;
    use crate::{ArrayLen, IntoArrayData};

    #[test]
    fn test_project() {
        let xs = PrimitiveArray::new(buffer![0i64, 1, 2, 3, 4], Validity::NonNullable);
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::NonNullable),
        );
        let zs = BoolArray::from_iter([true, true, true, false, false]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into(), "ys".into(), "zs".into()]),
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

        let bools = BoolArray::try_from(struct_b.maybe_null_field_by_idx(0).unwrap()).unwrap();
        assert_eq!(
            bools.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, true, true, false, false]
        );

        let prims = PrimitiveArray::try_from(struct_b.maybe_null_field_by_idx(1).unwrap()).unwrap();
        assert_eq!(prims.as_slice::<i64>(), [0i64, 1, 2, 3, 4]);
    }
}
