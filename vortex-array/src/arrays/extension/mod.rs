use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, ExtID};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityChild,
    ValidityVTableFromChild, VisitorVTable,
};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EncodingId, EncodingRef,
    IntoArray, vtable,
};

mod compute;
mod serde;

vtable!(Extension);

impl VTable for ExtensionVTable {
    type Array = ExtensionArray;
    type Encoding = ExtensionEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.ext")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ExtensionEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ExtensionEncoding;

#[derive(Clone, Debug)]
pub struct ExtensionArray {
    dtype: DType,
    storage: ArrayRef,
    stats_set: ArrayStats,
}

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: ArrayRef) -> Self {
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );
        Self {
            dtype: DType::Extension(ext_dtype),
            storage,
            stats_set: ArrayStats::default(),
        }
    }

    pub fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext) = &self.dtype else {
            unreachable!("ExtensionArray: dtype must be an ExtDType")
        };
        ext
    }

    pub fn storage(&self) -> &ArrayRef {
        &self.storage
    }

    #[allow(dead_code)]
    #[inline]
    pub fn id(&self) -> &ExtID {
        self.ext_dtype().id()
    }
}

impl ArrayVTable<ExtensionVTable> for ExtensionVTable {
    fn len(array: &ExtensionArray) -> usize {
        array.storage.len()
    }

    fn dtype(array: &ExtensionArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ExtensionArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl ValidityChild<ExtensionVTable> for ExtensionVTable {
    fn validity_child(array: &ExtensionArray) -> &dyn Array {
        array.storage.as_ref()
    }
}

impl CanonicalVTable<ExtensionVTable> for ExtensionVTable {
    fn canonicalize(array: &ExtensionArray) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(array.clone()))
    }
}

impl OperationsVTable<ExtensionVTable> for ExtensionVTable {
    fn slice(array: &ExtensionArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ExtensionArray::new(
            array.ext_dtype().clone(),
            array.storage().slice(start, stop)?,
        )
        .into_array())
    }

    fn scalar_at(array: &ExtensionArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension(
            array.ext_dtype().clone(),
            array.storage().scalar_at(index)?,
        ))
    }
}

impl VisitorVTable<ExtensionVTable> for ExtensionVTable {
    fn visit_buffers(_array: &ExtensionArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExtensionArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", array.storage.as_ref());
    }
}
