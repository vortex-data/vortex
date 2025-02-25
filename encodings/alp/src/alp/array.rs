use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::{StatisticsVTable, VTableRef};
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayExt, ArrayImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariantsImpl, Canonical, Encoding, EncodingId, SerdeMetadata,
    encoding_ids,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::alp::serde::ALPMetadata;
use crate::alp::{Exponents, alp_encode, decompress};

#[derive(Clone, Debug)]
pub struct ALPArray {
    dtype: DType,
    encoded: ArrayRef,
    exponents: Exponents,
    patches: Option<Patches>,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct ALPEncoding;
impl Encoding for ALPEncoding {
    const ID: EncodingId = EncodingId::new("vortex.alp", encoding_ids::ALP);
    type Array = ALPArray;
    type Metadata = SerdeMetadata<ALPMetadata>;
}

impl ALPArray {
    // TODO(ngates): remove try_new and panic on wrong DType?
    pub fn try_new(
        encoded: ArrayRef,
        exponents: Exponents,
        patches: Option<Patches>,
    ) -> VortexResult<Self> {
        let dtype = match encoded.dtype() {
            DType::Primitive(PType::I32, nullability) => DType::Primitive(PType::F32, *nullability),
            DType::Primitive(PType::I64, nullability) => DType::Primitive(PType::F64, *nullability),
            d => vortex_bail!(MismatchedTypes: "int32 or int64", d),
        };
        Ok(Self {
            dtype,
            encoded,
            exponents,
            patches,
            stats_set: Default::default(),
        })
    }

    pub fn encode(array: ArrayRef) -> VortexResult<ArrayRef> {
        if let Some(parray) = array.as_opt::<PrimitiveArray>() {
            Ok(alp_encode(parray)?.into_array())
        } else {
            vortex_bail!("ALP can only encode primitive arrays");
        }
    }

    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }

    #[inline]
    pub fn exponents(&self) -> Exponents {
        self.exponents
    }

    pub fn patches(&self) -> Option<&Patches> {
        self.patches.as_ref()
    }
}

impl ArrayImpl for ALPArray {
    type Encoding = ALPEncoding;

    fn _len(&self) -> usize {
        self.encoded.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&ALPEncoding)
    }
}

impl ArrayCanonicalImpl for ALPArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        decompress(self).map(Canonical::Primitive)
    }
}

impl ArrayStatisticsImpl for ALPArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

impl ArrayValidityImpl for ALPArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.encoded.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.encoded.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.encoded.all_invalid()
    }

    fn _valid_count(&self) -> VortexResult<usize> {
        self.encoded.valid_count()
    }

    fn _invalid_count(&self) -> VortexResult<usize> {
        self.encoded.invalid_count()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.encoded.validity_mask()
    }
}

impl ArrayVariantsImpl for ALPArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for ALPArray {}

impl StatisticsVTable<&ALPArray> for ALPEncoding {}

#[cfg(test)]
mod tests {
    use vortex_array::SerdeMetadata;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::Exponents;
    use crate::alp::serde::ALPMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alp_metadata() {
        check_metadata(
            "alp.metadata",
            SerdeMetadata(ALPMetadata {
                patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
                exponents: Exponents {
                    e: u8::MAX,
                    f: u8::MAX,
                },
            }),
        );
    }
}
