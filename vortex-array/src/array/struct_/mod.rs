use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_dtype::field::Field;
use vortex_dtype::{DType, FieldName, FieldNames, StructDType};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult};

use crate::encoding::ids;
use crate::stats::{ArrayStatistics, Stat, StatisticsVTable, StatsSet};
use crate::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use crate::variants::{ArrayVariants, StructArrayTrait};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData,
    IntoCanonical,
};

mod compute;

impl_encoding!("vortex.struct", ids::STRUCT, Struct);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructMetadata {
    validity: ValidityMetadata,
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
            self.field(idx).unwrap_or_else(|| {
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
            DType::Struct(StructDType::new(names, field_dtypes), nullability),
            length,
            StructMetadata {
                validity: validity_metadata,
            },
            children.into(),
            StatsSet::default(),
        )
    }

    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayData)]) -> VortexResult<Self> {
        let names: Vec<FieldName> = items
            .iter()
            .map(|(name, _)| FieldName::from(name.as_ref()))
            .collect();
        let fields: Vec<ArrayData> = items.iter().map(|(_, array)| array.clone()).collect();
        let len = fields
            .first()
            .map(|f| f.len())
            .ok_or_else(|| vortex_err!("StructArray cannot be constructed from an empty slice of arrays because the length is unspecified"))?;

        Self::try_new(FieldNames::from(names), fields, len, Validity::NonNullable)
    }

    // TODO(aduffy): Add equivalent function to support field masks for nested column access.
    /// Return a new StructArray with the given projection applied.
    ///
    /// Projection does not copy data arrays. Projection is defined by an ordinal array slice
    /// which specifies the new ordering of columns in the struct. The projection can be used to
    /// perform column re-ordering, deletion, or duplication at a logical level, without any data
    /// copying.
    #[allow(clippy::same_name_method)]
    pub fn project(&self, projection: &[Field]) -> VortexResult<Self> {
        let mut children = Vec::with_capacity(projection.len());
        let mut names = Vec::with_capacity(projection.len());

        for field in projection.iter() {
            let idx = match field {
                Field::Name(n) => self
                    .names()
                    .iter()
                    .position(|name| name.as_ref() == n)
                    .ok_or_else(|| vortex_err!("Unknown field {n}"))?,
                Field::Index(i) => *i,
            };

            names.push(self.names()[idx].clone());
            children.push(
                self.field(idx)
                    .ok_or_else(|| vortex_err!(OutOfBounds: idx, 0, self.dtypes().len()))?,
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

impl ArrayTrait for StructArray {}

impl ArrayVariants for StructArray {
    fn as_struct_array(&self) -> Option<&dyn StructArrayTrait> {
        Some(self)
    }
}

impl StructArrayTrait for StructArray {
    fn field(&self, idx: usize) -> Option<ArrayData> {
        self.dtypes().get(idx).map(|dtype| {
            self.as_ref()
                .child(idx, dtype, self.len())
                .unwrap_or_else(|e| vortex_panic!(e, "StructArray: field {} not found", idx))
        })
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData> {
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
    fn is_valid(&self, array: &StructArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &StructArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<StructArray> for StructEncoding {
    fn accept(&self, array: &StructArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        for (idx, name) in array.names().iter().enumerate() {
            let child = array
                .field(idx)
                .ok_or_else(|| vortex_err!(OutOfBounds: idx, 0, array.nfields()))?;
            visitor.visit_child(&format!("\"{}\"", name), &child)?;
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
    use vortex_dtype::field::Field;
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
        let xs = PrimitiveArray::from_vec(vec![0i64, 1, 2, 3, 4], Validity::NonNullable);
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
            .project(&[Field::from(2usize), Field::from(0)])
            .unwrap();
        assert_eq!(
            struct_b.names().as_ref(),
            [FieldName::from("zs"), FieldName::from("xs")],
        );

        assert_eq!(struct_b.len(), 5);

        let bools = BoolArray::try_from(struct_b.field(0).unwrap()).unwrap();
        assert_eq!(
            bools.boolean_buffer().iter().collect::<Vec<_>>(),
            vec![true, true, true, false, false]
        );

        let prims = PrimitiveArray::try_from(struct_b.field(1).unwrap()).unwrap();
        assert_eq!(prims.maybe_null_slice::<i64>(), [0i64, 1, 2, 3, 4]);
    }
}
