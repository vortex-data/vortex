use std::fmt::Debug;

pub use compress::*;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::{EncodingVTable, VTableRef};
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, Encoding, EncodingId,
};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::r#for::serde::ScalarValueMetadata;

mod compress;
mod compute;
mod serde;

#[derive(Clone, Debug)]
pub struct FoRArray {
    encoded: ArrayRef,
    reference: Scalar,
    stats_set: ArrayStats,
}

pub struct FoREncoding;
impl Encoding for FoREncoding {
    type Array = FoRArray;
    type Metadata = ScalarValueMetadata;
}

impl EncodingVTable for FoREncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("fastlanes.for")
    }
}

impl FoRArray {
    pub fn try_new(encoded: ArrayRef, reference: Scalar) -> VortexResult<Self> {
        if reference.is_null() {
            vortex_bail!("Reference value cannot be null");
        }
        let reference = reference.cast(
            &reference
                .dtype()
                .with_nullability(encoded.dtype().nullability()),
        )?;
        Ok(Self {
            encoded,
            reference,
            stats_set: Default::default(),
        })
    }

    #[inline]
    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }

    #[inline]
    pub fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }
}

impl ArrayImpl for FoRArray {
    type Encoding = FoREncoding;

    fn _len(&self) -> usize {
        self.encoded().len()
    }

    fn _dtype(&self) -> &DType {
        self.reference_scalar().dtype()
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&FoREncoding)
    }
}

impl ArrayCanonicalImpl for FoRArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        decompress(self).map(Canonical::Primitive)
    }
}

impl ArrayStatisticsImpl for FoRArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for FoRArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.encoded().is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.encoded().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.encoded().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.encoded().validity_mask()
    }
}

impl ArrayVariantsImpl for FoRArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for FoRArray {}
