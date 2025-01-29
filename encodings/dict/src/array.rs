use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::compute::{scalar_at, take};
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{CanonicalVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use vortex_array::{
    encoding_ids, impl_encoding, ArrayData, Canonical, IntoArrayData, IntoArrayVariant,
    IntoCanonical, SerdeMetadata,
};
use vortex_dtype::{match_each_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

impl_encoding!(
    "vortex.dict",
    encoding_ids::DICT,
    Dict,
    SerdeMetadata<DictMetadata>
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictMetadata {
    codes_ptype: PType,
    values_len: usize, // TODO(ngates): make this a u32
}

impl DictArray {
    pub fn try_new(codes: ArrayData, values: ArrayData) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() || codes.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable unsigned int", codes.dtype());
        }
        Self::try_from_parts(
            values.dtype().clone(),
            codes.len(),
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::try_from(codes.dtype())
                    .vortex_expect("codes dtype must be uint"),
                values_len: values.len(),
            }),
            None,
            Some([codes, values].into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn codes(&self) -> ArrayData {
        self.as_ref()
            .child(0, &DType::from(self.metadata().codes_ptype), self.len())
            .vortex_expect("DictArray is missing its codes child array")
    }

    #[inline]
    pub fn values(&self) -> ArrayData {
        self.as_ref()
            .child(1, self.dtype(), self.metadata().values_len)
            .vortex_expect("DictArray is missing its values child array")
    }
}

impl ValidateVTable<DictArray> for DictEncoding {}

impl CanonicalVTable<DictArray> for DictEncoding {
    fn into_canonical(&self, array: DictArray) -> VortexResult<Canonical> {
        match array.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: ArrayData = array.values().into_canonical()?.into_array();
                take(canonical_values, array.codes())?.into_canonical()
            }
            // Non-string case: take and then canonicalize
            _ => take(array.values(), array.codes())?.into_canonical(),
        }
    }
}

impl ValidityVTable<DictArray> for DictEncoding {
    fn is_valid(&self, array: &DictArray, index: usize) -> VortexResult<bool> {
        let values_index = scalar_at(array.codes(), index)
            .unwrap_or_else(|err| {
                vortex_panic!(err, "Failed to get index {} from DictArray codes", index)
            })
            .as_ref()
            .try_into()
            .vortex_expect("Failed to convert dictionary code to usize");
        array.values().is_valid(values_index)
    }

    fn logical_validity(&self, array: &DictArray) -> VortexResult<Mask> {
        if array.dtype().is_nullable() {
            let primitive_codes = array.codes().into_primitive()?;
            match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                let is_valid = primitive_codes
                    .as_slice::<$P>();
                let is_valid_buffer = BooleanBuffer::collect_bool(is_valid.len(), |idx| {
                    is_valid[idx] != 0
                });
                Ok(Mask::from_buffer(is_valid_buffer))
            })
        } else {
            Ok(Mask::AllTrue(array.len()))
        }
    }
}

impl VisitorVTable<DictArray> for DictEncoding {
    fn accept(&self, array: &DictArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("values", &array.values())?;
        visitor.visit_child("codes", &array.codes())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::SerdeMetadata;
    use vortex_dtype::PType;

    use crate::DictMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        check_metadata(
            "dict.metadata",
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::U64,
                values_len: usize::MAX,
            }),
        );
    }
}
