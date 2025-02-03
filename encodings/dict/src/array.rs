use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use num_traits::NumCast;
use serde::{Deserialize, Serialize};
use vortex_array::compute::{scalar_at, take};
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{CanonicalVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use vortex_array::{
    encoding_ids, impl_encoding, Array, Canonical, IntoArray, IntoArrayVariant, IntoCanonical,
    SerdeMetadata,
};
use vortex_dtype::{match_each_integer_ptype, DType, NativePType, PType};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult};
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
    // TODO(robert): This goes away once we move validity around
    single_null_code: Option<u64>,
}

impl DictArray {
    /// Construct new Dictionary encoded array.
    ///
    /// # Note:
    ///
    /// Assumes that only null_code argument is invalid in values array
    pub fn try_new(codes: Array, values: Array, null_code: Option<u64>) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() || codes.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable unsigned int", codes.dtype());
        }
        if !values.dtype().is_nullable() && null_code.is_some() {
            vortex_bail!("Can only provide null code when values are nullable")
        }
        if let Some(null_code) = null_code {
            if values.is_valid(usize::try_from(null_code)?)? {
                vortex_bail!("Null code {null_code} points to a valid value")
            }
        }

        Self::try_from_parts(
            values.dtype().clone(),
            codes.len(),
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::try_from(codes.dtype())
                    .vortex_expect("codes dtype must be uint"),
                values_len: values.len(),
                single_null_code: null_code,
            }),
            None,
            Some([codes, values].into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn codes(&self) -> Array {
        self.as_ref()
            .child(0, &DType::from(self.metadata().codes_ptype), self.len())
            .vortex_expect("DictArray is missing its codes child array")
    }

    #[inline]
    pub fn values(&self) -> Array {
        self.as_ref()
            .child(1, self.dtype(), self.metadata().values_len)
            .vortex_expect("DictArray is missing its values child array")
    }

    #[inline]
    pub fn null_code(&self) -> Option<u64> {
        self.metadata().single_null_code
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
                let canonical_values: Array = array.values().into_canonical()?.into_array();
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

    fn all_valid(&self, array: &DictArray) -> VortexResult<bool> {
        // If the values are all valid, then the dictionary must be all valid
        if array.values().all_valid()? {
            return Ok(true);
        }

        if let Some(null_code) = array.null_code() {
            let primitive_codes = array.codes().into_primitive()?;
            match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                all_valid(primitive_codes.as_slice::<$P>(), null_code)
            })
        } else {
            Ok(true)
        }
    }

    fn validity_mask(&self, array: &DictArray) -> VortexResult<Mask> {
        if let Some(null_code) = array.null_code() {
            let primitive_codes = array.codes().into_primitive()?;
            match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                validity_mask(primitive_codes.as_slice::<$P>(), null_code)
            })
        } else {
            Ok(Mask::AllTrue(array.len()))
        }
    }
}

fn validity_mask<T: NativePType + NumCast>(codes: &[T], null_code: u64) -> VortexResult<Mask> {
    let null_code = T::from_u64(null_code)
        .ok_or_else(|| vortex_err!("Can't cast u64 null code to code type"))?;
    let is_valid_buffer = BooleanBuffer::collect_bool(codes.len(), |idx| codes[idx] != null_code);
    Ok(Mask::from_buffer(is_valid_buffer))
}

fn all_valid<T: NativePType + NumCast>(codes: &[T], null_code: u64) -> VortexResult<bool> {
    let null_code = T::from_u64(null_code)
        .ok_or_else(|| vortex_err!("Can't cast u64 null code to code type"))?;
    for code in codes.iter().copied() {
        if code == null_code {
            return Ok(false);
        }
    }
    Ok(true)
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
                single_null_code: Some(u64::MAX),
            }),
        );
    }
}
