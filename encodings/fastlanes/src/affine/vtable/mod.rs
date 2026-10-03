// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::require_child;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::AffineData;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::affine::array::AffineArrayExt;
use crate::affine::array::AffineArraySlotsExt;
use crate::affine::array::AffineSlots;
use crate::affine::array::AffineSlotsView;
use crate::affine::array::affine_decompress::decompress;
use crate::affine::array::num_chunks;
use crate::affine::plugin::affine_id;
use crate::affine::vtable::rules::PARENT_RULES;

mod operations;
mod rules;
mod slice;
mod validity;

/// An [`Affine`]-encoded Vortex array.
pub type AffineArray = Array<Affine>;

impl ArrayHash for AffineData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
        self.slope_shift.hash(state);
    }
}

impl ArrayEq for AffineData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset && self.slope_shift == other.slope_shift
    }
}

impl VTable for Affine {
    type TypedArrayData = AffineData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        affine_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = AffineSlotsView::from_slots(slots);
        validate_parts(&slots, data.offset, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("AffineArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        AffineSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("Affine serialization requires AffinePlugin")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("Affine deserialization requires AffinePlugin")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = if array.references().is::<Constant>() {
            array
        } else {
            require_child!(array, array.references(), AffineSlots::REFERENCES => Primitive)
        };
        let array = if array.scales().is::<Constant>() {
            array
        } else {
            require_child!(array, array.scales(), AffineSlots::SCALES => Primitive)
        };
        let array = if array.slopes().is::<Constant>() {
            array
        } else {
            require_child!(array, array.slopes(), AffineSlots::SLOPES => Primitive)
        };
        // Dictionary-encoded residuals decode fused with the model, reading the dictionary's own
        // children, so the dictionary itself is left unexecuted.
        if array.encoded().is::<Dict>() {
            return Ok(ExecutionResult::done(decompress(&array, ctx)?.into_array()));
        }
        // A slice of a bit-packed child with patches stays lazy until executed. Step it to the
        // sliced bit-packed array, so the fused unpack below still applies.
        let slice_of_bitpacked = array
            .encoded()
            .as_opt::<Slice>()
            .is_some_and(|slice| slice.child().is::<BitPacked>());
        let array = if slice_of_bitpacked {
            require_child!(array, array.encoded(), AffineSlots::ENCODED => BitPacked)
        } else {
            array
        };
        // The fused unpack reads a bit-packed child's buffers directly when its chunks line up.
        let fused = array
            .encoded()
            .as_opt::<BitPacked>()
            .is_some_and(|bp| bp.offset() == array.offset());
        let array = if fused {
            array
        } else {
            require_child!(array, array.encoded(), AffineSlots::ENCODED => Primitive)
        };
        Ok(ExecutionResult::done(decompress(&array, ctx)?.into_array()))
    }
}

#[derive(Clone, Debug)]
pub struct Affine;

impl Affine {
    /// Construct an Affine array from its residuals and per-chunk parameters.
    ///
    /// `encoded` holds the residuals, either of the array's integer type or of a narrower unsigned
    /// type. `references` and `scales` must be non-nullable arrays of the array's integer type,
    /// and `slopes` a non-nullable `i64` array, each with one entry for every chunk spanned by
    /// `offset + encoded.len()` elements. `offset` is the position of the first element within the
    /// first chunk.
    pub fn try_new(
        encoded: ArrayRef,
        references: ArrayRef,
        scales: ArrayRef,
        slopes: ArrayRef,
        offset: u16,
        slope_shift: u8,
    ) -> VortexResult<AffineArray> {
        let dtype = references.dtype().with_nullability(encoded.dtype().nullability());
        let len = encoded.len();
        let data = AffineData::try_new(offset, slope_shift)?;
        let slots = smallvec![Some(encoded), Some(references), Some(scales), Some(slopes)];
        Array::try_from_parts(ArrayParts::new(Affine, dtype, len, data).with_slots(slots))
    }
}

fn validate_parts(
    slots: &AffineSlotsView<'_>,
    offset: u16,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(dtype.is_int(), "Affine requires an integer dtype, got {dtype}");
    let encoded_dtype = slots.encoded.dtype();
    vortex_ensure!(
        encoded_dtype == dtype
            || (encoded_dtype.is_unsigned_int()
                && encoded_dtype.nullability() == dtype.nullability()
                && encoded_dtype.as_ptype().byte_width() <= dtype.as_ptype().byte_width()),
        "Affine encoded dtype must be {dtype} or a no wider unsigned integer, got {encoded_dtype}"
    );
    vortex_ensure!(
        slots.encoded.len() == len,
        "Affine encoded length mismatch: expected {len}, got {}",
        slots.encoded.len()
    );
    let num_chunks = num_chunks(offset, len);
    let param_dtype = dtype.as_nonnullable();
    let slope_dtype = DType::Primitive(PType::I64, false.into());
    for (name, child, expected) in [
        ("references", slots.references, &param_dtype),
        ("scales", slots.scales, &param_dtype),
        ("slopes", slots.slopes, &slope_dtype),
    ] {
        vortex_ensure!(
            child.dtype() == expected,
            "Affine {name} dtype mismatch: expected {expected}, got {}",
            child.dtype()
        );
        vortex_ensure!(
            child.len() == num_chunks,
            "Affine expects {num_chunks} {name}, got {}",
            child.len()
        );
    }
    Ok(())
}
