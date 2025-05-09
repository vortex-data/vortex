use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::stats::{ArrayStats, StatsSet, StatsSetRef};
use crate::vtable::{VTable, ValidityVTable};
use crate::{Array, ArrayRef, Encoding, EncodingRef, vtable};

mod canonical;
mod compute;
mod serde;
mod variants;

vtable!(Constant);

#[derive(Clone, Debug)]
pub struct ConstantArray {
    scalar: Scalar,
    len: usize,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct ConstantEncoding;

impl VTable for ConstantVTable {
    type Array = ConstantArray;
    type Encoding = ConstantEncoding;
    type CanonicalVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> ArcRef<str> {
        ArcRef::new_ref("vortex.constant")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        ArcRef::new_ref(&ConstantEncoding)
    }

    fn len(array: &Self::Array) -> usize {
        array.len
    }

    fn dtype(array: &Self::Array) -> &DType {
        array.scalar.dtype()
    }

    fn stats(array: &Self::Array) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array)
    }

    fn slice(array: &Self::Array, start: usize, stop: usize) -> VortexResult<Self::Array> {
        Ok(ConstantArray::new(array.scalar.clone(), stop - start))
    }

    fn scalar_at(array: &Self::Array, _index: usize) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }

    fn with_children(array: &Self::Array, _children: &[ArrayRef]) -> VortexResult<Self::Array> {
        Ok(array.clone())
    }
}

impl ConstantArray {
    pub fn new<S>(scalar: S, len: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        let stats = StatsSet::constant(scalar.clone(), len);
        Self {
            scalar,
            len,
            stats_set: ArrayStats::from(stats),
        }
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }
}

impl ValidityVTable<ConstantVTable> for ConstantVTable {
    fn is_valid(array: &<ConstantVTable as VTable>::Array, _index: usize) -> VortexResult<bool> {
        Ok(!array.scalar().is_null())
    }

    fn all_valid(array: &<ConstantVTable as VTable>::Array) -> VortexResult<bool> {
        Ok(!array.scalar().is_null())
    }

    fn all_invalid(array: &<ConstantVTable as VTable>::Array) -> VortexResult<bool> {
        Ok(array.scalar().is_null())
    }

    fn validity_mask(array: &<ConstantVTable as VTable>::Array) -> VortexResult<Mask> {
        Ok(match array.scalar().is_null() {
            true => Mask::AllFalse(array.len()),
            false => Mask::AllTrue(array.len()),
        })
    }
}
