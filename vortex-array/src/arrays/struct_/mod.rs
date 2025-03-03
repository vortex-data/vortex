use std::fmt::Debug;
use std::sync::Arc;

use vortex_dtype::{DType, FieldName, FieldNames, StructDType};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;

use crate::array::{ArrayCanonicalImpl, ArrayValidityImpl};
use crate::stats::{ArrayStats, Precision, Stat, StatsSet, StatsSetRef};
use crate::validity::Validity;
use crate::variants::StructArrayTrait;
use crate::vtable::{EncodingVTable, StatisticsVTable, VTableRef};
use crate::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayVariantsImpl, Canonical, EmptyMetadata,
    Encoding, EncodingId,
};
mod compute;
mod serde;

#[derive(Clone, Debug)]
pub struct StructArray {
    len: usize,
    dtype: DType,
    fields: Vec<ArrayRef>,
    validity: Validity,
    stats_set: ArrayStats,
}

pub struct StructEncoding;
impl Encoding for StructEncoding {
    type Array = StructArray;
    type Metadata = EmptyMetadata;
}

impl EncodingVTable for StructEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.struct")
    }
}

impl StructArray {
    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    pub fn fields(&self) -> &[ArrayRef] {
        &self.fields
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
                    field.len()
                );
            }
        }

        let field_dtypes: Vec<_> = fields.iter().map(|d| d.dtype()).cloned().collect();
        let dtype = DType::Struct(Arc::new(StructDType::new(names, field_dtypes)), nullability);

        Ok(Self {
            len: length,
            dtype,
            fields,
            validity,
            stats_set: Default::default(),
        })
    }

    pub fn from_fields<N: AsRef<str>>(items: &[(N, ArrayRef)]) -> VortexResult<Self> {
        let names = items.iter().map(|(name, _)| FieldName::from(name.as_ref()));
        let fields: Vec<ArrayRef> = items.iter().map(|(_, array)| array.to_array()).collect();
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
                    .vortex_expect("never out of bounds"),
            );
        }

        StructArray::try_new(
            FieldNames::from(names.as_slice()),
            children,
            self.len(),
            self.validity().clone(),
        )
    }
}

impl ArrayImpl for StructArray {
    type Encoding = StructEncoding;

    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&StructEncoding)
    }
}

impl ArrayStatisticsImpl for StructArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayVariantsImpl for StructArray {
    fn _as_struct_typed(&self) -> Option<&dyn StructArrayTrait> {
        Some(self)
    }
}

impl StructArrayTrait for StructArray {
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<ArrayRef> {
        Ok(self.fields[idx].clone())
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayRef> {
        self.project(projection).map(|a| a.into_array())
    }
}

impl ArrayCanonicalImpl for StructArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Struct(self.clone()))
    }
}

impl ArrayValidityImpl for StructArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_logical(self.len())
    }
}

impl StatisticsVTable<&StructArray> for StructEncoding {
    fn compute_statistics(&self, array: &StructArray, stat: Stat) -> VortexResult<StatsSet> {
        Ok(match stat {
            Stat::NullCount => StatsSet::of(
                stat,
                Precision::exact(array.validity().null_count(array.len())?),
            ),
            _ => StatsSet::default(),
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability};

    use crate::ArrayExt;
    use crate::array::Array;
    use crate::arrays::BoolArray;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::arrays::struct_::StructArray;
    use crate::arrays::varbin::VarBinArray;
    use crate::validity::Validity;
    use crate::variants::StructArrayTrait;

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

        let bools = struct_b.maybe_null_field_by_idx(0).unwrap();
        assert_eq!(
            bools
                .as_::<BoolArray>()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, true, true, false, false]
        );

        let prims = struct_b.maybe_null_field_by_idx(1).unwrap();
        assert_eq!(
            prims.as_::<PrimitiveArray>().as_slice::<i64>(),
            [0i64, 1, 2, 3, 4]
        );
    }
}
