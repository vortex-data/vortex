use std::fmt::Debug;

use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, Encoding, ProstMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::alp::serde::ALPMetadata;
use crate::alp::{Exponents, decompress};

#[derive(Clone, Debug)]
pub struct ALPArray {
    encoded: ArrayRef,
    patches: Option<Patches>,
    dtype: DType,
    exponents: Exponents,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct ALPEncoding;
impl Encoding for ALPEncoding {
    type Array = ALPArray;
    type Metadata = ProstMetadata<ALPMetadata>;
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

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let encoded = children[0].clone();

        let patches = self.patches().map(|existing| {
            let indices = children[1].clone();
            let values = children[2].clone();
            Patches::new(existing.array_len(), existing.offset(), indices, values)
        });

        ALPArray::try_new(encoded, self.exponents(), patches)
    }
}

impl ArrayCanonicalImpl for ALPArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        decompress(self).map(Canonical::Primitive)
    }
}

impl ArrayStatisticsImpl for ALPArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
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
