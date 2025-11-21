// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::fmt::{Debug, Formatter};
use std::hash::Hash;

pub use compress::*;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, EncodeVTable, NotSupported, VTable, ValidityChild,
    ValidityVTableFromChild, VisitorVTable,
};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef, Canonical,
    DeserializeMetadata, EncodingId, EncodingRef, Precision, SerializeMetadata, vtable,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

mod compress;
mod compute;
mod ops;

vtable!(FoR);

impl VTable for FoRVTable {
    type Array = FoRArray;
    type Encoding = FoREncoding;
    type Metadata = ScalarValueMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.for")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FoREncoding.as_ref())
    }

    fn metadata(array: &FoRArray) -> VortexResult<Self::Metadata> {
        Ok(ScalarValueMetadata(
            array.reference_scalar().value().clone(),
        ))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        ScalarValueMetadata::deserialize(buffer)
    }

    fn build(
        _encoding: &FoREncoding,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FoRArray> {
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let encoded = children.get(0, dtype, len)?;
        let reference = Scalar::new(dtype.clone(), metadata.0.clone());

        FoRArray::try_new(encoded, reference)
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

    pub(crate) unsafe fn new_unchecked(encoded: ArrayRef, reference: Scalar) -> Self {
        Self {
            encoded,
            reference,
            stats_set: Default::default(),
        }
    }

    #[inline]
    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
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

    fn array_hash<H: std::hash::Hasher>(array: &FoRArray, state: &mut H, precision: Precision) {
        array.encoded.array_hash(state, precision);
        array.reference.hash(state);
    }

    fn array_eq(array: &FoRArray, other: &FoRArray, precision: Precision) -> bool {
        array.encoded.array_eq(&other.encoded, precision) && array.reference == other.reference
    }
}

impl ValidityChild<FoRVTable> for FoRVTable {
    fn validity_child(array: &FoRArray) -> &dyn Array {
        array.encoded().as_ref()
    }
}

impl CanonicalVTable<FoRVTable> for FoRVTable {
    fn canonicalize(array: &FoRArray) -> Canonical {
        Canonical::Primitive(decompress(array))
    }
}

impl EncodeVTable<FoRVTable> for FoRVTable {
    fn encode(
        _encoding: &FoREncoding,
        canonical: &Canonical,
        _like: Option<&FoRArray>,
    ) -> VortexResult<Option<FoRArray>> {
        let parray = canonical.clone().into_primitive();
        Ok(Some(FoRArray::encode(parray)?))
    }
}

impl VisitorVTable<FoRVTable> for FoRVTable {
    fn visit_buffers(_array: &FoRArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children<'a>(array: &'a FoRArray, visitor: &mut dyn ArrayChildVisitor<'a>) {
        visitor.visit_child("encoded", array.encoded())
    }
}

#[derive(Clone)]
pub struct ScalarValueMetadata(pub ScalarValue);

impl SerializeMetadata for ScalarValueMetadata {
    fn serialize(self) -> Vec<u8> {
        self.0.to_protobytes()
    }
}

impl DeserializeMetadata for ScalarValueMetadata {
    type Output = ScalarValueMetadata;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let scalar_value = ScalarValue::from_protobytes(metadata)?;
        Ok(ScalarValueMetadata(scalar_value))
    }
}

impl Debug for ScalarValueMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}
