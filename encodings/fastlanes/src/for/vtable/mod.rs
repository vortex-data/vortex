// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{
    ArrayId, ArrayVTable, ArrayVTableExt, NotSupported, VTable, ValidityVTableFromChild,
};
use vortex_array::{DeserializeMetadata, SerializeMetadata, vtable};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::FoRArray;

mod array;
mod canonical;
mod encode;
mod operations;
mod operator;
mod validity;
mod visitor;

vtable!(FoR);

impl VTable for FoRVTable {
    type Array = FoRArray;

    type Metadata = ScalarValueMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("fastlanes.for")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        FoRVTable.as_vtable()
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
        &self,
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
pub struct FoRVTable;

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
