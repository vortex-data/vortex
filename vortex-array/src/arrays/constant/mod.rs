use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::stats::{ArrayStats, StatsSet, StatsSetRef};
use crate::vtable::{
    ArrayVTable, NotSupported, OperationsVTable, VTable, ValidityVTable, VisitorVTable,
};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, EncodingId, EncodingRef, IntoArray, vtable,
};

mod canonical;
mod compute;
mod encode;
mod serde;

vtable!(Constant);

#[derive(Clone, Debug)]
pub struct ConstantArray {
    scalar: Scalar,
    len: usize,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ConstantEncoding;

impl VTable for ConstantVTable {
    type Array = ConstantArray;
    type Encoding = ConstantEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    // TODO(ngates): implement a compute kernel for elementwise operations
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.constant")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ConstantEncoding.as_ref())
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

impl ArrayVTable<ConstantVTable> for ConstantVTable {
    fn len(array: &ConstantArray) -> usize {
        array.len
    }

    fn dtype(array: &ConstantArray) -> &DType {
        array.scalar.dtype()
    }

    fn stats(array: &ConstantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl OperationsVTable<ConstantVTable> for ConstantVTable {
    fn slice(array: &ConstantArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar.clone(), stop - start).into_array())
    }

    fn scalar_at(array: &ConstantArray, _index: usize) -> VortexResult<Scalar> {
        Ok(array.scalar.clone())
    }
}

impl ValidityVTable<ConstantVTable> for ConstantVTable {
    fn is_valid(array: &ConstantArray, _index: usize) -> VortexResult<bool> {
        Ok(!array.scalar().is_null())
    }

    fn all_valid(array: &ConstantArray) -> VortexResult<bool> {
        Ok(!array.scalar().is_null())
    }

    fn all_invalid(array: &ConstantArray) -> VortexResult<bool> {
        Ok(array.scalar().is_null())
    }

    fn validity_mask(array: &ConstantArray) -> VortexResult<Mask> {
        Ok(match array.scalar().is_null() {
            true => Mask::AllFalse(array.len()),
            false => Mask::AllTrue(array.len()),
        })
    }
}

impl VisitorVTable<ConstantVTable> for ConstantVTable {
    fn visit_buffers(array: &ConstantArray, visitor: &mut dyn ArrayBufferVisitor) {
        let buffer = array
            .scalar
            .value()
            .to_protobytes::<ByteBufferMut>()
            .freeze();
        visitor.visit_buffer(&buffer);
    }

    fn visit_children(_array: &ConstantArray, _visitor: &mut dyn ArrayChildVisitor) {}
}
