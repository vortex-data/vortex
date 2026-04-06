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

use crate::encodings::turboquant::TurboQuantArrayExt;
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
        let len = norms.len();
        TurboQuantData::validate(&dtype, &codes, &norms, &centroids, &rotation_signs)?;
        let ext = TurboQuant::validate_dtype(&dtype)?;
        let dimension = tensor_list_size(ext)?;
        let bit_width = if centroids.is_empty() {
            0
        } else {
            u8::try_from(centroids.len().trailing_zeros())
                .map_err(|_| vortex_err!("centroids bit_width does not fit in u8"))?
        };
        let data = TurboQuantData::try_new(dimension, bit_width)?;
        let parts = ArrayParts::new(TurboQuant, dtype, len, data).with_slots(
            TurboQuantData::make_slots(codes, norms, centroids, rotation_signs),
        );

        Array::try_from_parts(parts)
    }
}

vtable!(TurboQuant, TurboQuant, TurboQuantData);

impl ArrayHash for TurboQuantData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.dimension.hash(state);
        self.bit_width.hash(state);
    }
}

impl ArrayEq for TurboQuantData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.dimension == other.dimension && self.bit_width == other.bit_width
    }
}

impl VTable for TurboQuant {
    type ArrayData = TurboQuantData;
    type OperationsVTable = TurboQuant;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(
            slots.len(),
            Slot::COUNT,
            "TurboQuantArray expects {} slots, got {}",
            Slot::COUNT,
            slots.len()
        );

        let codes = slots[Slot::Codes as usize]
            .as_ref()
            .ok_or_else(|| vortex_err!("TurboQuantArray codes slot"))?;
        let norms = slots[Slot::Norms as usize]
            .as_ref()
            .ok_or_else(|| vortex_err!("TurboQuantArray norms slot"))?;
        let centroids = slots[Slot::Centroids as usize]
            .as_ref()
            .ok_or_else(|| vortex_err!("TurboQuantArray centroids slot"))?;
        let rotation_signs = slots[Slot::RotationSigns as usize]
            .as_ref()
            .ok_or_else(|| vortex_err!("TurboQuantArray rotation_signs slot"))?;

        TurboQuantData::validate(dtype, codes, norms, centroids, rotation_signs)?;
        vortex_ensure_eq!(
            norms.len(),
            len,
            "TurboQuant norms length does not match outer length",
        );
        vortex_ensure_eq!(
            data.dimension,
            tensor_list_size(TurboQuant::validate_dtype(dtype)?)?
        );

        let expected_bit_width = if centroids.is_empty() {
            0
        } else {
            u8::try_from(centroids.len().trailing_zeros())
                .map_err(|_| vortex_err!("centroids bit_width does not fit in u8"))?
        };
        vortex_ensure_eq!(
            data.bit_width,
            expected_bit_width,
            "TurboQuant bit_width does not match centroids slot",
        );

        Ok(())
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
    ) -> VortexResult<ArrayParts<Self>> {
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

        Ok(ArrayParts::new(
            TurboQuant,
            dtype.clone(),
            len,
            TurboQuantData {
                dimension,
                bit_width,
            },
        )
        .with_slots(TurboQuantData::make_slots(
            codes_array,
            norms_array,
            centroids,
            rotation_signs,
        )))
    }

    fn slot_name(_array: ArrayView<Self>, idx: usize) -> String {
        Slot::from_index(idx).name().to_string()
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
    fn validity_child(array: ArrayView<'_, TurboQuant>) -> ArrayRef {
        array.norms().clone()
    }
}
