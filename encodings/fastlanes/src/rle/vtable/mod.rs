// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::sync::Arc;

use prost::Message;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChildSliceHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::RLEArray;
use crate::rle::array::rle_decompress::rle_decompress;
use crate::rle::kernel::PARENT_KERNELS;
use crate::rle::vtable::rules::RULES;

mod operations;
mod rules;
mod validity;

vtable!(RLE);

#[derive(Clone, prost::Message)]
pub struct RLEMetadata {
    #[prost(uint64, tag = "1")]
    pub values_len: u64,
    #[prost(uint64, tag = "2")]
    pub indices_len: u64,
    #[prost(enumeration = "PType", tag = "3")]
    pub indices_ptype: i32,
    #[prost(uint64, tag = "4")]
    pub values_idx_offsets_len: u64,
    #[prost(enumeration = "PType", tag = "5")]
    pub values_idx_offsets_ptype: i32,
    #[prost(uint64, tag = "6", default = "0")]
    pub offset: u64,
}

impl VTable for RLE {
    type Array = RLEArray;

    type Metadata = ProstMetadata<RLEMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChildSliceHelper;

    fn vtable(_array: &Self::Array) -> &Self {
        &RLE
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &RLEArray) -> usize {
        array.len()
    }

    fn dtype(array: &RLEArray) -> &DType {
        array.dtype()
    }

    fn stats(array: &RLEArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &RLEArray, state: &mut H, precision: Precision) {
        array.dtype().hash(state);
        array.values().array_hash(state, precision);
        array.indices().array_hash(state, precision);
        array.values_idx_offsets().array_hash(state, precision);
        array.offset().hash(state);
        array.len().hash(state);
    }

    fn array_eq(array: &RLEArray, other: &RLEArray, precision: Precision) -> bool {
        array.dtype() == other.dtype()
            && array.values().array_eq(other.values(), precision)
            && array.indices().array_eq(other.indices(), precision)
            && array
                .values_idx_offsets()
                .array_eq(other.values_idx_offsets(), precision)
            && array.offset() == other.offset()
            && array.len() == other.len()
    }

    fn nbuffers(_array: &RLEArray) -> usize {
        0
    }

    fn buffer(_array: &RLEArray, idx: usize) -> BufferHandle {
        vortex_panic!("RLEArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &RLEArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &RLEArray) -> usize {
        3
    }

    fn child(array: &RLEArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.values().clone(),
            1 => array.indices().clone(),
            2 => array.values_idx_offsets().clone(),
            _ => vortex_panic!("RLEArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &RLEArray, idx: usize) -> String {
        match idx {
            0 => "values".to_string(),
            1 => "indices".to_string(),
            2 => "values_idx_offsets".to_string(),
            _ => vortex_panic!("RLEArray child name index {idx} out of bounds"),
        }
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // RLEArray children order (from visit_children):
        // 1. values
        // 2. indices
        // 3. values_idx_offsets

        vortex_ensure!(
            children.len() == 3,
            "Expected 3 children for RLE encoding, got {}",
            children.len()
        );

        array.values = children[0].clone();
        array.indices = children[1].clone();
        array.values_idx_offsets = children[2].clone();

        Ok(())
    }

    fn metadata(array: &RLEArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(RLEMetadata {
            values_len: array.values().len() as u64,
            indices_len: array.indices().len() as u64,
            indices_ptype: PType::try_from(array.indices().dtype())? as i32,
            values_idx_offsets_len: array.values_idx_offsets().len() as u64,
            values_idx_offsets_ptype: PType::try_from(array.values_idx_offsets().dtype())? as i32,
            offset: array.offset() as u64,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(RLEMetadata::decode(bytes)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<RLEArray> {
        let metadata = &metadata.0;
        let values = children.get(
            0,
            &DType::Primitive(dtype.as_ptype(), Nullability::NonNullable),
            usize::try_from(metadata.values_len)?,
        )?;

        let indices = children.get(
            1,
            &DType::Primitive(metadata.indices_ptype(), dtype.nullability()),
            usize::try_from(metadata.indices_len)?,
        )?;

        let values_idx_offsets = children.get(
            2,
            &DType::Primitive(
                metadata.values_idx_offsets_ptype(),
                Nullability::NonNullable,
            ),
            usize::try_from(metadata.values_idx_offsets_len)?,
        )?;

        RLEArray::try_new(
            values,
            indices,
            values_idx_offsets,
            metadata.offset as usize,
            len,
        )
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            rle_decompress(&array, ctx)?.into_array(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct RLE;

impl RLE {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.rle");
}

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::ProstMetadata;
    use super::RLEMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_rle_metadata() {
        check_metadata(
            "rle.metadata",
            ProstMetadata(RLEMetadata {
                values_len: u64::MAX,
                indices_len: u64::MAX,
                indices_ptype: i32::MAX,
                values_idx_offsets_len: u64::MAX,
                values_idx_offsets_ptype: i32::MAX,
                offset: u64::MAX,
            }),
        );
    }
}
