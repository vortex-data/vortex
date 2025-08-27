use super::*;

use vortex_dtype::{DType, Nullability};
use vortex_scalar::Scalar;

use crate::arrays::FixedSizeListArray;
use crate::stats::StatsSetRef;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use crate::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};

vtable!(FixedSizeList);

#[derive(Clone, Debug)]
pub struct FixedSizeListEncoding;

impl VTable for FixedSizeListVTable {
    type Array = FixedSizeListArray;
    type Encoding = FixedSizeListEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type PipelineVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.list")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FixedSizeListEncoding.as_ref())
    }
}

impl ArrayVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn len(array: &FixedSizeListArray) -> usize {
        array.len
    }

    fn dtype(array: &FixedSizeListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FixedSizeListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl OperationsVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn slice(array: &FixedSizeListArray, start: usize, stop: usize) -> ArrayRef {
        let len = start - stop;
        let list_size = array.list_size() as usize;

        FixedSizeListArray::new(
            array.values().slice(start * list_size, stop * list_size),
            array.list_size(),
            array.validity().slice(start, stop),
            len,
        )
        .into_array()
    }

    fn scalar_at(array: &FixedSizeListArray, index: usize) -> Scalar {
        let list = array.fixed_size_list_at(index);
        let children_elements = (0..list.len()).map(|i| list.scalar_at(i)).collect();

        Scalar::fixed_size_list(
            array.dtype().clone(),
            children_elements,
            array.dtype.nullability(),
        )
    }
}

impl CanonicalVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn canonicalize(array: &FixedSizeListArray) -> VortexResult<Canonical> {
        Ok(Canonical::FixedSizeList(array.clone()))
    }
}

impl ValidityHelper for FixedSizeListArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
