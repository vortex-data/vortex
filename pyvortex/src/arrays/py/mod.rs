mod array;
mod encoding;

pub use array::*;
pub use encoding::*;
use vortex::vtable::{NotSupported, VTable};
use vortex::{EncodingId, EncodingRef, vtable};

vtable!(Python);

impl VTable for PythonVTable {
    type Array = PythonArray;
    type Encoding = PythonEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        todo!()
    }

    fn encoding(array: &Self::Array) -> EncodingRef {
        PythonArray::encoding(array)
    }
}
