// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EmptyArrayData;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::arrays::ConstantArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::with_empty_buffers;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::l2_denorm::execute::denormalize;
use crate::encodings::l2_denorm::rules::RULES;
use crate::encodings::l2_denorm::validate::validate_l2_denorm_children;
use crate::encodings::l2_denorm::validate::validate_l2_normalized_rows_against_norms;
use crate::utils::validate_tensor_float_input;

/// An [`L2Denorm`]-encoded Vortex array.
pub type L2DenormArray = Array<L2Denorm>;

/// The norm-split encoding for tensor-like columns.
///
/// Row `i` decodes to `normalized[i] * norms[i]`, which is exactly the original tensor row when
/// `normalized[i]` is unit-norm. The encoding covers both logical tensor dtypes reachable through
/// [`AnyTensor`]: `Vector` and `FixedShapeTensor`.
///
/// # Invariants
///
/// Every [`L2DenormArray`] structurally guarantees, via [`VTable::validate`]:
///
/// - `normalized` is a tensor-like extension array with a float element type.
/// - `norms` is a primitive column whose ptype equals the tensor element ptype.
/// - both children have the array's length.
/// - the array dtype is `normalized.dtype().union_nullability(norms.nullability())`.
///
/// On top of that, [`try_new`](Self::try_new) enforces the semantic invariants that make the split
/// lossless:
///
/// - every valid row of `normalized` has L2 norm `1.0` or `0.0`, within the tolerance implied by
///   the element precision.
/// - every stored norm is non-negative.
/// - a stored norm of `0.0` is paired with an all-zero normalized row.
///
/// # Lossy normalized children
///
/// [`new_unchecked`](Self::new_unchecked) deliberately skips the semantic scan so that
/// `normalized` may be an *approximation* of the unit-norm direction, such as a quantized child.
/// The stored norms stay authoritative in that case, and the read-through rules in
/// [`L2Norm`], [`InnerProduct`], and [`CosineSimilarity`] are defined against the stored children
/// rather than against decoded coordinates. Those operators may therefore return slightly
/// different answers than fully decoding both operands and recomputing. That difference is the
/// storage contract, not a separate lossy-compute mode.
///
/// [`AnyTensor`]: crate::matcher::AnyTensor
/// [`L2Norm`]: crate::scalar_fns::l2_norm::L2Norm
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
#[derive(Clone, Debug)]
pub struct L2Denorm;

/// The two child arrays of an [`L2DenormArray`].
#[array_slots(L2Denorm)]
pub struct L2DenormSlots {
    /// The unit-norm (or zero) direction of each row, as a tensor-like extension array.
    #[slot(0)]
    pub normalized: ArrayRef,

    /// The authoritative L2 norm of each row, as a primitive float column.
    #[slot(1)]
    pub norms: ArrayRef,
}

impl L2Denorm {
    /// Builds an [`L2DenormArray`], validating that `normalized` really is row-wise L2-normalized
    /// against `norms`.
    ///
    /// This is the constructor for exact norm splits. It scans both children, so it costs
    /// `O(len * list_size)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children are structurally incompatible, or if they violate any of
    /// the semantic invariants listed on [`L2Denorm`].
    pub fn try_new(
        normalized: ArrayRef,
        norms: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<L2DenormArray> {
        let len = normalized.len();
        let dtype = normalized
            .dtype()
            .union_nullability(norms.dtype().nullability());
        let slots = L2DenormSlots { normalized, norms }.into_slots();

        // Structural validation has to come first: the row scan walks both children in lockstep
        // and assumes they are a matching-length tensor/float pair.
        let denorm = Array::try_from_parts(
            ArrayParts::new(L2Denorm, dtype, len, EmptyArrayData).with_slots(slots),
        )?;
        validate_l2_normalized_rows_against_norms(denorm.normalized(), Some(denorm.norms()), ctx)?;

        Ok(denorm)
    }

    /// Builds an [`L2DenormArray`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must uphold the structural invariants listed on [`L2Denorm`]. In particular,
    /// both children must have the same length, `normalized` must be a float tensor, and `norms`
    /// must be a primitive column with the same element ptype.
    ///
    /// This does not check the unit-norm relationship. Violating it can produce wrong answers but
    /// not memory unsafety.
    pub unsafe fn new_unchecked(normalized: ArrayRef, norms: ArrayRef) -> L2DenormArray {
        let len = normalized.len();
        let dtype = normalized
            .dtype()
            .union_nullability(norms.dtype().nullability());
        let slots = L2DenormSlots { normalized, norms }.into_slots();

        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(L2Denorm, dtype, len, EmptyArrayData).with_slots(slots),
            )
        }
    }
}

/// Metadata for a serialized [`L2DenormArray`]: its children's nullabilities.
///
/// The parent dtype supplies the tensor shape and element ptype. Its nullability is the union of
/// the children, so it cannot identify which child is nullable.
#[derive(Clone, prost::Message)]
pub struct L2DenormMetadata {
    /// Whether the `normalized` child is nullable.
    #[prost(bool, tag = "1")]
    pub normalized_is_nullable: bool,

    /// Whether the `norms` child is nullable.
    #[prost(bool, tag = "2")]
    pub norms_is_nullable: bool,
}

impl VTable for L2Denorm {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.tensor.l2_denorm");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = L2DenormSlotsView::from_slots(slots);

        validate_l2_denorm_children(slots.normalized, slots.norms, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("L2DenormArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("L2DenormArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            L2DenormMetadata {
                normalized_is_nullable: array.normalized().dtype().is_nullable(),
                norms_is_nullable: array.norms().dtype().is_nullable(),
            }
            .encode_to_vec(),
        ))
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
        let metadata = L2DenormMetadata::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode L2DenormMetadata: {e}"))?;

        let element_ptype = validate_tensor_float_input(dtype)?.element_ptype();
        let normalized_dtype = dtype.with_nullability(metadata.normalized_is_nullable.into());
        let norms_dtype = DType::Primitive(element_ptype, metadata.norms_is_nullable.into());

        let normalized = children.get(0, &normalized_dtype, len)?;
        let norms = children.get(1, &norms_dtype, len)?;
        let slots = L2DenormSlots { normalized, norms }.into_slots();

        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        L2DenormSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let dtype = array.dtype().clone();
        let slots = array.slots_view();

        denormalize(slots.normalized, slots.norms, array.len(), dtype, ctx)
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl ValidityVTable<L2Denorm> for L2Denorm {
    fn validity(array: ArrayView<'_, L2Denorm>) -> VortexResult<Validity> {
        array
            .normalized()
            .validity()?
            .and(array.norms().validity()?)
    }
}

impl OperationsVTable<L2Denorm> for L2Denorm {
    fn scalar_at(
        array: ArrayView<'_, L2Denorm>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Denormalize a single row rather than the whole column: both children are collapsed to
        // one-row constants, which also lets the constant-norms fast path do the multiply.
        let normalized = array.normalized().execute_scalar(index, ctx)?;
        let norms = array.norms().execute_scalar(index, ctx)?;

        let row = denormalize(
            &ConstantArray::new(normalized, 1).into_array(),
            &ConstantArray::new(norms, 1).into_array(),
            1,
            array.dtype().clone(),
            ctx,
        )?;

        row.execute_scalar(0, ctx)
    }
}
