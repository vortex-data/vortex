// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VTable implementation for TurboQuant encoding.

use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
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
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::array::Slot;
use crate::array::TurboQuant;
use crate::array::TurboQuantArray;
use crate::array::TurboQuantMetadata;
use crate::decompress::execute_decompress;

impl VTable for TurboQuant {
    type Array = TurboQuantArray;
    type Metadata = ProstMetadata<TurboQuantMetadata>;
    type OperationsVTable = TurboQuant;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &TurboQuant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &TurboQuantArray) -> usize {
        array.norms().len()
    }

    fn dtype(array: &TurboQuantArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &TurboQuantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &TurboQuantArray,
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

    fn array_eq(array: &TurboQuantArray, other: &TurboQuantArray, precision: Precision) -> bool {
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

    fn nbuffers(_array: &TurboQuantArray) -> usize {
        0
    }

    fn buffer(_array: &TurboQuantArray, idx: usize) -> BufferHandle {
        vortex_panic!("TurboQuantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &TurboQuantArray, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: &TurboQuantArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &TurboQuantArray, idx: usize) -> String {
        Slot::from_index(idx).name().to_string()
    }

    fn with_slots(array: &mut TurboQuantArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == Slot::COUNT,
            "TurboQuantArray expects {} slots, got {}",
            Slot::COUNT,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn metadata(array: &TurboQuantArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(TurboQuantMetadata {
            dimension: array.dimension,
            bit_width: array.bit_width as u32,
            has_qjl: array.has_qjl(),
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

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<TurboQuantArray> {
        let bit_width = u8::try_from(metadata.bit_width)?;
        let padded_dim = metadata.dimension.next_power_of_two() as usize;
        let num_centroids = 1usize << bit_width;

        let u8_nn = DType::Primitive(PType::U8, Nullability::NonNullable);
        let f32_nn = DType::Primitive(PType::F32, Nullability::NonNullable);
        let codes_dtype = DType::FixedSizeList(
            Arc::new(u8_nn.clone()),
            padded_dim as u32,
            Nullability::NonNullable,
        );
        let codes = children.get(0, &codes_dtype, len)?;

        let norms = children.get(1, &f32_nn, len)?;
        let centroids = children.get(2, &f32_nn, num_centroids)?;

        let signs_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let rotation_signs = children.get(3, &signs_dtype, 3 * padded_dim)?;

        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Norms as usize] = Some(norms);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);

        if metadata.has_qjl {
            let qjl_signs_dtype =
                DType::FixedSizeList(Arc::new(u8_nn), padded_dim as u32, Nullability::NonNullable);
            slots[Slot::QjlSigns as usize] = Some(children.get(4, &qjl_signs_dtype, len)?);
            slots[Slot::QjlResidualNorms as usize] = Some(children.get(5, &f32_nn, len)?);
            slots[Slot::QjlRotationSigns as usize] =
                Some(children.get(6, &signs_dtype, 3 * padded_dim)?);
        }

        Ok(TurboQuantArray {
            dtype: dtype.clone(),
            slots,
            dimension: metadata.dimension,
            bit_width,
            stats_set: Default::default(),
        })
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::compute::rules::RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::compute::rules::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = Arc::try_unwrap(array)
            .map(|a| a.into_inner())
            .unwrap_or_else(|arc| arc.as_ref().deref().clone());
        Ok(ExecutionResult::done(execute_decompress(inner, ctx)?))
    }
}

impl ValidityChild<TurboQuant> for TurboQuant {
    fn validity_child(array: &TurboQuantArray) -> &ArrayRef {
        array.codes()
    }
}
