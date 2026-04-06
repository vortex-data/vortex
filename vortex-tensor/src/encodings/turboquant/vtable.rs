// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! VTable implementation for TurboQuant encoding.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::Precision;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::encodings::turboquant::TurboQuantData;
use crate::encodings::turboquant::array::slots::Slot;
use crate::encodings::turboquant::compute::rules::PARENT_KERNELS;
use crate::encodings::turboquant::compute::rules::RULES;
use crate::encodings::turboquant::decompress::execute_decompress;
use crate::utils::tensor_element_ptype;
use crate::utils::tensor_list_size;
use crate::vector::Vector;

/// Encoding marker type for TurboQuant.
#[derive(Clone, Debug)]
pub struct TurboQuant;

impl TurboQuant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant");

    /// Minimum vector dimension for TurboQuant encoding.
    pub const MIN_DIMENSION: u32 = 128;

    /// Validates that `dtype` is a [`Vector`](crate::vector::Vector) extension type with
    /// dimension >= [`MIN_DIMENSION`](Self::MIN_DIMENSION).
    ///
    /// Returns the validated [`ExtDTypeRef`] on success, which can be used to extract the
    /// element ptype and list size.
    pub fn validate_dtype(dtype: &DType) -> VortexResult<&ExtDTypeRef> {
        let ext = dtype
            .as_extension_opt()
            .filter(|e| e.is::<Vector>())
            .ok_or_else(|| {
                vortex_err!("TurboQuant dtype must be a Vector extension type, got {dtype}")
            })?;

        let dimension = tensor_list_size(ext)?;
        vortex_ensure!(
            dimension >= Self::MIN_DIMENSION,
            "TurboQuant requires dimension >= {}, got {dimension}",
            Self::MIN_DIMENSION
        );

        Ok(ext)
    }

    /// Creates a new [`TurboQuantArray`].
    ///
    /// Internally calls [`TurboQuantData::try_new`].
    pub fn try_new_array(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
    ) -> VortexResult<TurboQuantArray> {
        let data = TurboQuantData::try_new(&dtype, codes, norms, centroids, rotation_signs)?;

        let parts = ArrayParts::new(TurboQuant, dtype, data.norms().len(), data);

        Array::try_from_parts(parts)
    }
}

vtable!(TurboQuant, TurboQuant, TurboQuantData);

impl VTable for TurboQuant {
    type ArrayData = TurboQuantData;
    type OperationsVTable = TurboQuant;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, data: &Self::ArrayData, dtype: &DType, len: usize) -> VortexResult<()> {
        let ext = dtype
            .as_extension_opt()
            .filter(|e| e.is::<Vector>())
            .ok_or_else(|| {
                vortex_err!("TurboQuant dtype must be a Vector extension type, got {dtype}")
            })?;

        let dimension = tensor_list_size(ext)?;
        vortex_ensure!(
            dimension >= Self::MIN_DIMENSION,
            "TurboQuant requires dimension >= {}, got {dimension}",
            Self::MIN_DIMENSION
        );

        vortex_ensure_eq!(data.dimension(), dimension);

        // TODO(connor): In the future, we may not need to validate `len` on the array data because
        // the child arrays will be located somewhere else.
        // bit_width == 0 is only valid for degenerate (empty) arrays. A non-empty array with
        // bit_width == 0 would have zero centroids while codes reference centroid indices.
        vortex_ensure!(
            data.bit_width > 0 || len == 0,
            "bit_width == 0 is only valid for empty arrays, got len={len}"
        );

        Ok(())
    }

    fn array_hash<H: Hasher>(array: &TurboQuantData, state: &mut H, precision: Precision) {
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
        array.dimension == other.dimension
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

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![array.bit_width]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<Self::ArrayData> {
        vortex_ensure_eq!(
            metadata.len(),
            1,
            "TurboQuant metadata must be exactly 1 byte, got {}",
            metadata.len()
        );
        vortex_ensure!(
            metadata[0] <= 8,
            "bit_width is expected to be between 0 and 8, got {}",
            metadata[0]
        );

        let bit_width = metadata[0];

        // bit_width == 0 is only valid for degenerate (empty) arrays. A non-empty array with
        // bit_width == 0 would have zero centroids while codes reference centroid indices.
        vortex_ensure!(
            bit_width > 0 || len == 0,
            "bit_width == 0 is only valid for empty arrays, got len={len}"
        );

        // Validate and derive dimension and element ptype from the Vector extension dtype.
        let ext = TurboQuant::validate_dtype(dtype)?;
        let dimension = tensor_list_size(ext)?;
        let element_ptype = tensor_element_ptype(ext)?;

        let padded_dim = dimension.next_power_of_two();

        // Get the codes array (indices into the codebook). Codes are always non-nullable;
        // null vectors are represented by all-zero codes with a null norm.
        let codes_ptype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let codes_dtype =
            DType::FixedSizeList(Arc::new(codes_ptype), padded_dim, Nullability::NonNullable);
        let codes_array = children.get(0, &codes_dtype, len)?;

        // Get the L2 norms array. Norms carry the validity of the entire TurboQuant array:
        // null vectors have null norms.
        let norms_dtype = DType::Primitive(element_ptype, dtype.nullability());
        let norms_array = children.get(1, &norms_dtype, len)?;

        // Get the centroids array (codebook).
        let num_centroids = if bit_width == 0 {
            0 // A degenerate TQ array.
        } else {
            1usize << bit_width
        };
        let centroids_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let centroids = children.get(2, &centroids_dtype, num_centroids)?;

        // Get the rotation array.
        let signs_len = if len == 0 { 0 } else { 3 * padded_dim as usize };
        let signs_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let rotation_signs = children.get(3, &signs_dtype, signs_len)?;

        Ok(TurboQuantData {
            slots: vec![
                Some(codes_array),
                Some(norms_array),
                Some(centroids),
                Some(rotation_signs),
            ],
            dimension,
            bit_width,
        })
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

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(execute_decompress(array, ctx)?))
    }

    fn execute_parent(
        array: ArrayView<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: ArrayView<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl ValidityChild<TurboQuant> for TurboQuant {
    fn validity_child(array: &TurboQuantData) -> &ArrayRef {
        array.norms()
    }
}
