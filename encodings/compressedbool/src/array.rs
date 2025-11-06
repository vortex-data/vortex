// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

use vortex_array::arrays::{BoolArray, PrimitiveArray};
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use vortex_array::{
    ArrayEq, ArrayHash, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, Precision,
    ToCanonical, vtable,
};
use vortex_buffer::BitBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

vtable!(CompressedBool);

impl VTable for CompressedBoolVTable {
    type Array = CompressedBoolArray;
    type Encoding = CompressedBoolEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.compressedbool")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(CompressedBoolEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct CompressedBoolArray {
    dtype: DType,
    compressed_buffer: ArrayRef,
    validity: Validity,
    stats_set: ArrayStats,
    bit_offset: usize,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct CompressedBoolEncoding;

impl CompressedBoolArray {
    pub fn try_new(
        compressed_buffer: ArrayRef,
        validity: Validity,
        bit_offset: usize,
        len: usize,
    ) -> VortexResult<Self> {
        if compressed_buffer.dtype() != &DType::Primitive(PType::U8, Nullability::NonNullable) {
            vortex_bail!("must be non-nullable u8: {}", compressed_buffer.dtype());
        }

        if let Some(vlen) = validity.maybe_len()
            && len != vlen
        {
            vortex_bail!(
                "bit length ({}) does not match validity length ({})",
                len,
                vlen
            );
        }

        let n_bytes = compressed_buffer.len();
        if (bit_offset + len).div_ceil(8) != n_bytes {
            vortex_bail!(
                "compressed_buffer length ({}) does not match bit length ({})",
                n_bytes,
                len
            );
        }

        Ok(Self {
            dtype: DType::Bool(validity.nullability()),
            compressed_buffer,
            validity,
            stats_set: Default::default(),
            bit_offset,
            len,
        })
    }

    // TODO(ngates): deprecate construction from vec
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> Self {
        let len = data.len();
        let validity = validity.into();
        let buffer = BoolArray::from_iter(data.into_iter())
            .into_bit_buffer()
            .into_inner();

        Self::try_new(
            PrimitiveArray::new(buffer, Validity::NonNullable).into_array(),
            validity,
            0,
            len,
        )
        .expect("type is right; length could be wrong though")
    }

    pub fn bit_offset(&self) -> usize {
        self.bit_offset
    }

    pub fn compressed_buffer(&self) -> &ArrayRef {
        &self.compressed_buffer
    }
}

impl ValidityHelper for CompressedBoolArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl ArrayVTable<CompressedBoolVTable> for CompressedBoolVTable {
    fn len(array: &CompressedBoolArray) -> usize {
        array.len
    }

    fn dtype(array: &CompressedBoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &CompressedBoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &CompressedBoolArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.compressed_buffer.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(
        array: &CompressedBoolArray,
        other: &CompressedBoolArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array
                .compressed_buffer
                .array_eq(&other.compressed_buffer, precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}

impl CanonicalVTable<CompressedBoolVTable> for CompressedBoolVTable {
    fn canonicalize(array: &CompressedBoolArray) -> Canonical {
        let buffer = array.compressed_buffer().to_primitive();
        assert_eq!(buffer.ptype(), PType::U8);
        assert!(!buffer.dtype().is_nullable());
        let boolean_buffer = BitBuffer::new(buffer.into_byte_buffer(), array.len());
        let validity = array.validity().clone();
        Canonical::Bool(BoolArray::from_bit_buffer(boolean_buffer, validity))
    }
}

impl OperationsVTable<CompressedBoolVTable> for CompressedBoolVTable {
    fn slice(array: &CompressedBoolArray, range: Range<usize>) -> ArrayRef {
        let bit_len = range.len();
        let start = array.bit_offset + range.start;
        let end = array.bit_offset + range.end;
        let byte_start = start / 8;
        let byte_end = end.div_ceil(8);
        let bit_offset = start % 8;

        CompressedBoolArray::try_new(
            array.compressed_buffer().slice(byte_start..byte_end),
            array.validity().slice(range),
            bit_offset,
            bit_len,
        )
        .expect("must be ok")
        .into_array()
    }

    fn scalar_at(array: &CompressedBoolArray, index: usize) -> Scalar {
        let index = index + array.bit_offset;
        let scalar = array.scalar_at(index / 8);
        let value = scalar
            .as_primitive()
            .typed_value::<u8>()
            .vortex_expect("compressed buffer must be non-nullable");
        let bit = value & (1 << (index % 8)) != 0;
        Scalar::bool(bit, array.dtype().nullability())
    }
}

impl From<Vec<bool>> for CompressedBoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::from_vec(value, Validity::AllValid)
    }
}

impl From<Vec<Option<bool>>> for CompressedBoolArray {
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

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = CompressedBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx));
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = CompressedBoolArray::from(v);
        assert!(arr.is_valid(0));
        assert!(!arr.is_valid(1));
        assert!(arr.is_valid(2));
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = CompressedBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx));
        }
        assert_eq!(arr.len(), 2);
    }
}
