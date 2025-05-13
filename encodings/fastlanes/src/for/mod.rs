use std::fmt::Debug;

pub use compress::*;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityChild, ValidityVTableFromChild,
};
use vortex_array::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, vtable};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

mod compress;
mod compute;
mod ops;
mod serde;

vtable!(FoR);

impl VTable for FoRVTable {
    type Array = FoRArray;
    type Encoding = FoREncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.for")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FoREncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct FoRArray {
    encoded: ArrayRef,
    reference: Scalar,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct FoREncoding;

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
    pub fn ptype(&self) -> PType {
        self.dtype().to_ptype()
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

impl ArrayVTable<FoRVTable> for FoRVTable {
    fn len(array: &FoRArray) -> usize {
        array.encoded().len()
    }

    fn dtype(array: &FoRArray) -> &DType {
        array.reference_scalar().dtype()
    }

    fn stats(array: &FoRArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl ValidityChild<FoRVTable> for FoRVTable {
    fn validity_child(array: &FoRArray) -> &dyn Array {
        array.encoded().as_ref()
    }
}

impl CanonicalVTable<FoRVTable> for FoRVTable {
    fn canonicalize(array: &FoRArray) -> VortexResult<Canonical> {
        decompress(array).map(Canonical::Primitive)
    }
}
