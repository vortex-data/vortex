// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Range;

use vortex_array::arrays::BoolArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};
use vortex_buffer::{BitBuffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::vortex_panic;
use vortex_scalar::Scalar;

vtable!(ByteBool);

impl VTable for ByteBoolVTable {
    type Array = ByteBoolArray;
    type Encoding = ByteBoolEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.bytebool")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ByteBoolEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ByteBoolArray {
    dtype: DType,
    buffer: ByteBuffer,
    validity: Validity,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ByteBoolEncoding;

impl ByteBoolArray {
    pub fn new(buffer: ByteBuffer, validity: Validity) -> Self {
        let length = buffer.len();
        if let Some(vlen) = validity.maybe_len()
            && length != vlen
        {
            vortex_panic!(
                "Buffer length ({}) does not match validity length ({})",
                length,
                vlen
            );
        }
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: Default::default(),
        }
    }

    // TODO(ngates): deprecate construction from vec
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> Self {
        let validity = validity.into();
        // SAFETY: we are transmuting a Vec<bool> into a Vec<u8>
        let data: Vec<u8> = unsafe { std::mem::transmute(data) };
        Self::new(ByteBuffer::from(data), validity)
    }

    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    pub fn as_slice(&self) -> &[bool] {
        // Safety: The internal buffer contains byte-sized bools
        unsafe { std::mem::transmute(self.buffer().as_slice()) }
    }
}

impl ValidityHelper for ByteBoolArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<ByteBoolVTable> for ByteBoolVTable {
    fn len(array: &ByteBoolArray) -> usize {
        array.buffer.len()
    }

    fn dtype(array: &ByteBoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ByteBoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ByteBoolVTable> for ByteBoolVTable {
    fn canonicalize(array: &ByteBoolArray) -> Canonical {
        let boolean_buffer = BitBuffer::from(array.as_slice());
        let validity = array.validity().clone();
        Canonical::Bool(BoolArray::from_bit_buffer(boolean_buffer, validity))
    }
}

impl OperationsVTable<ByteBoolVTable> for ByteBoolVTable {
    fn slice(array: &ByteBoolArray, range: Range<usize>) -> ArrayRef {
        ByteBoolArray::new(
            array.buffer().slice(range.clone()),
            array.validity().slice(range),
        )
        .into_array()
    }

    fn scalar_at(array: &ByteBoolArray, index: usize) -> Scalar {
        Scalar::bool(array.buffer()[index] == 1, array.dtype().nullability())
    }
}

impl From<Vec<bool>> for ByteBoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::from_vec(value, Validity::AllValid)
    }
}

impl From<Vec<Option<bool>>> for ByteBoolArray {
    fn from(value: Vec<Option<bool>>) -> Self {
        let validity = Validity::from_iter(value.iter().map(|v| v.is_some()));

        // This doesn't reallocate, and the compiler even vectorizes it
        let data = value.into_iter().map(Option::unwrap_or_default).collect();

        Self::from_vec(data, validity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[cfg_attr(miri, ignore)]
    // #[test]
    // fn test_bytebool_metadata() {
    //     check_metadata(
    //         "bytebool.metadata",
    //         SerdeMetadata(ByteBoolMetadata {
    //             validity: ValidityMetadata::AllValid,
    //         }),
    //     );
    // }

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx));
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBoolArray::from(v);
        assert!(arr.is_valid(0));
        assert!(!arr.is_valid(1));
        assert!(arr.is_valid(2));
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx));
        }
        assert_eq!(arr.len(), 2);
    }
}
