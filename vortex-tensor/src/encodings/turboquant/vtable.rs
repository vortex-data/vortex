// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VTable implementation for TurboQuant encoding.

use std::hash::Hash;
use std::sync::Arc;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::encodings::turboquant::array::Slot;
use crate::encodings::turboquant::array::TurboQuant;
use crate::encodings::turboquant::array::TurboQuantData;
use crate::encodings::turboquant::array::TurboQuantMetadata;
use crate::encodings::turboquant::decompress::execute_decompress;

impl VTable for TurboQuant {
    type ArrayData = TurboQuantData;
    type Metadata = ProstMetadata<TurboQuantMetadata>;
    type OperationsVTable = TurboQuant;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &TurboQuant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &TurboQuantData) -> usize {
        array.norms().len()
    }

    fn dtype(array: &TurboQuantData) -> &DType {
        &array.dtype
    }

    fn stats(array: &TurboQuantData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &TurboQuantData,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.dimension.hash(state);
        array.bit_width.hash(state);
        for slot in &array.slots {
            slot.is_some().hash(state);
            if let Some(child) = slot {
                child.array_hash(state, precision);
            }
        }
    }

    fn array_eq(array: &TurboQuantData, other: &TurboQuantData, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.dimension == other.dimension
            && array.bit_width == other.bit_width
            && array.slots.len() == other.slots.len()
            && array
                .slots
                .iter()
                .zip(other.slots.iter())
                .all(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => a.array_eq(b, precision),
                    (None, None) => true,
                    _ => false,
                })
    }

    fn nbuffers(_array: ArrayView<Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<Self>, idx: usize) -> BufferHandle {
        vortex_panic!("TurboQuantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<Self>, idx: usize) -> String {
        Slot::from_index(idx).name().to_string()
    }

    fn with_slots(array: &mut TurboQuantData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == Slot::COUNT,
            "TurboQuantArray expects {} slots, got {}",
            Slot::COUNT,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn metadata(array: ArrayView<Self>) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(TurboQuantMetadata {
            dimension: array.dimension,
            bit_width: array.bit_width as u32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<TurboQuantMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<TurboQuantData> {
        let bit_width = u8::try_from(metadata.bit_width)?;
        let padded_dim = metadata.dimension.next_power_of_two() as usize;
        let num_centroids = 1usize << bit_width;

        let u8_nn = DType::Primitive(PType::U8, Nullability::NonNullable);
        let f32_nn = DType::Primitive(PType::F32, Nullability::NonNullable);
        let codes_dtype =
            DType::FixedSizeList(Arc::new(u8_nn), padded_dim as u32, Nullability::NonNullable);
        let codes = children.get(0, &codes_dtype, len)?;

        let norms = children.get(1, &f32_nn, len)?;
        let centroids = children.get(2, &f32_nn, num_centroids)?;

        let signs_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let rotation_signs = children.get(3, &signs_dtype, 3 * padded_dim)?;

        Ok(TurboQuantData {
            dtype: dtype.clone(),
            slots: vec![
                Some(codes),
                Some(norms),
                Some(centroids),
                Some(rotation_signs),
            ],
            dimension: metadata.dimension,
            bit_width,
            stats_set: Default::default(),
        })
    }

    fn reduce_parent(
        array: ArrayView<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::encodings::turboquant::compute::rules::RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::encodings::turboquant::compute::rules::PARENT_KERNELS
            .execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(execute_decompress(array, ctx)?))
    }
}

impl ValidityChild<TurboQuant> for TurboQuant {
    fn validity_child(array: &TurboQuantData) -> &ArrayRef {
        array.codes()
    }
}
