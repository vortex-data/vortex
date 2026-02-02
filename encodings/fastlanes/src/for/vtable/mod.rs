// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;

use crate::FoRArray;
use crate::r#for::array::for_decompress::decompress;
use crate::r#for::vtable::rules::PARENT_RULES;

mod array;
mod operations;
mod rules;
mod validity;
mod visitor;

vtable!(FoR);

impl VTable for FoRVTable {
    type Array = FoRArray;

    type Metadata = ScalarValueMetadata;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // FoRArray children order (from visit_children):
        // 1. encoded

        vortex_ensure!(
            children.len() == 1,
            "Expected 1 child for FoR encoding, got {}",
            children.len()
        );

        array.encoded = children[0].clone();

        Ok(())
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
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
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

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: Just slicing encoded data does not affect FOR.
        Ok(Some(unsafe {
            FoRArray::new_unchecked(
                array.encoded().slice(range)?,
                array.reference_scalar().clone(),
            )
            .into_array()
        }))
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        Ok(Canonical::Primitive(decompress(array, ctx)?))
    }
}

#[derive(Debug)]
pub struct FoRVTable;

impl FoRVTable {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.for");
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
