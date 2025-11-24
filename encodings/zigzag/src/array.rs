// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;

use vortex_array::serde::ArrayChildren;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayId, ArrayVTable, ArrayVTableExt, BaseArrayVTable, CanonicalVTable, EncodeVTable,
    NotSupported, OperationsVTable, VTable, ValidityChild, ValidityVTableFromChild, VisitorVTable,
};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef, Canonical,
    EmptyMetadata, IntoArray, Precision, ToCanonical, vtable,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;
use zigzag::ZigZag as ExternalZigZag;

use crate::compute::ZigZagEncoded;
use crate::{zigzag_decode, zigzag_encode};

vtable!(ZigZag);

impl VTable for ZigZagVTable {
    type Array = ZigZagArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.zigzag")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ZigZagVTable.as_vtable()
    }

    fn metadata(_array: &ZigZagArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZigZagArray> {
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = children.get(0, &encoded_type, len)?;
        ZigZagArray::try_new(encoded)
    }
}

#[derive(Clone, Debug)]
pub struct ZigZagArray {
    dtype: DType,
    encoded: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ZigZagVTable;

impl ZigZagArray {
    pub fn new(encoded: ArrayRef) -> Self {
        Self::try_new(encoded).vortex_expect("ZigZigArray new")
    }

    pub fn try_new(encoded: ArrayRef) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        Ok(Self {
            dtype,
            encoded,
            stats_set: Default::default(),
        })
    }

    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }
}

impl BaseArrayVTable<ZigZagVTable> for ZigZagVTable {
    fn len(array: &ZigZagArray) -> usize {
        array.encoded.len()
    }

    fn dtype(array: &ZigZagArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZigZagArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ZigZagArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.encoded.array_hash(state, precision);
    }

    fn array_eq(array: &ZigZagArray, other: &ZigZagArray, precision: Precision) -> bool {
        array.dtype == other.dtype && array.encoded.array_eq(&other.encoded, precision)
    }
}

impl CanonicalVTable<ZigZagVTable> for ZigZagVTable {
    fn canonicalize(array: &ZigZagArray) -> Canonical {
        Canonical::Primitive(zigzag_decode(array.encoded().to_primitive()))
    }
}

impl OperationsVTable<ZigZagVTable> for ZigZagVTable {
    fn slice(array: &ZigZagArray, range: Range<usize>) -> ArrayRef {
        ZigZagArray::new(array.encoded().slice(range)).into_array()
    }

    fn scalar_at(array: &ZigZagArray, index: usize) -> Scalar {
        let scalar = array.encoded().scalar_at(index);
        if scalar.is_null() {
            return scalar.reinterpret_cast(array.ptype());
        }

        let pscalar = scalar.as_primitive();
        match_each_unsigned_integer_ptype!(pscalar.ptype(), |P| {
            Scalar::primitive(
                <<P as ZigZagEncoded>::Int>::decode(
                    pscalar
                        .typed_value::<P>()
                        .vortex_expect("zigzag corruption"),
                ),
                array.dtype().nullability(),
            )
        })
    }
}

impl ValidityChild<ZigZagVTable> for ZigZagVTable {
    fn validity_child(array: &ZigZagArray) -> &dyn Array {
        array.encoded()
    }
}

impl EncodeVTable<ZigZagVTable> for ZigZagVTable {
    fn encode(
        encoding: &ZigZagVTable,
        canonical: &Canonical,
        _like: Option<&ZigZagArray>,
    ) -> VortexResult<Option<ZigZagArray>> {
        let parray = canonical.clone().into_primitive();

        if !parray.ptype().is_signed_int() {
            vortex_bail!(
                "only signed integers can be encoded into {}, got {}",
                encoding.id(),
                parray.ptype()
            )
        }

        Ok(Some(zigzag_encode(parray)?))
    }
}

impl VisitorVTable<ZigZagVTable> for ZigZagVTable {
    fn visit_buffers(_array: &ZigZagArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ZigZagArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_compute_statistics() {
        let array = buffer![1i32, -5i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        let canonical = array.to_canonical();
        let zigzag = ZigZagVTable
            .as_vtable()
            .encode(&canonical, None)
            .unwrap()
            .unwrap();

        assert_eq!(
            zigzag.statistics().compute_max::<i32>(),
            array.statistics().compute_max::<i32>()
        );
        assert_eq!(
            zigzag.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            zigzag.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );

        let sliced = zigzag.slice(0..2);
        let sliced = sliced.as_::<ZigZagVTable>();
        assert_eq!(sliced.scalar_at(sliced.len() - 1), Scalar::from(-5i32));

        assert_eq!(
            sliced.statistics().compute_min::<i32>(),
            array.statistics().compute_min::<i32>()
        );
        assert_eq!(
            sliced.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            sliced.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );
    }
}
