use std::fmt::Debug;

use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityChild, ValidityVTableFromChild,
};
use vortex_array::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, vtable};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::alp::{Exponents, decompress};

vtable!(ALP);

impl VTable for ALPVTable {
    type Array = ALPArray;
    type Encoding = ALPEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.alp")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ALPEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ALPArray {
    encoded: ArrayRef,
    patches: Option<Patches>,
    dtype: DType,
    exponents: Exponents,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ALPEncoding;

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

    pub fn ptype(&self) -> PType {
        self.dtype.to_ptype()
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

impl ValidityChild<ALPVTable> for ALPVTable {
    fn validity_child(array: &ALPArray) -> &dyn Array {
        array.encoded()
    }
}

impl ArrayVTable<ALPVTable> for ALPVTable {
    fn len(array: &ALPArray) -> usize {
        array.encoded.len()
    }

    fn dtype(array: &ALPArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ALPArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ALPVTable> for ALPVTable {
    fn canonicalize(array: &ALPArray) -> VortexResult<Canonical> {
        decompress(array).map(Canonical::Primitive)
    }
}
