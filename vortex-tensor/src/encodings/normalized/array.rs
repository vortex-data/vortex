// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EmptyArrayData;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_array::vtable::with_empty_buffers;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::normalized::execute::denormalize;
use crate::encodings::normalized::rules::RULES;
use crate::encodings::normalized::validate::validate_normalized_children;
use crate::encodings::normalized::validate::validate_normalized_rows;
use crate::types::unit_vector::AnyUnitVector;
use crate::types::unit_vector::UnitVector;
use crate::types::vector::AnyVector;
use crate::types::vector::Vector;
use crate::utils::validate_tensor_float_input;

/// A [`Normalized`]-encoded Vortex array.
pub type NormalizedArray = Array<Normalized>;

/// The norm-split encoding for tensor-like columns.
///
/// Row `i` decodes to `normalized[i] * norms[i]`. The encoding supports [`Vector`] and
/// [`FixedShapeTensor`] columns with float elements.
///
/// # Invariants
///
/// Every [`NormalizedArray`] has three slots.
///
/// - For a [`Vector`] parent, `normalized` is a non-nullable
///   [`UnitVector`]. A documented lossy transform may instead
///   erase the refinement and store an ordinary non-nullable Vector.
/// - For a [`FixedShapeTensor`] parent, `normalized` is the corresponding non-nullable tensor.
/// - `norms` is a non-nullable primitive column with the tensor element ptype.
/// - `validity` optionally contains the parent validity as a non-nullable boolean column.
///
/// Both data children have the array's length and element ptype. A missing validity slot means
/// either non-nullable or nullable-all-valid data, as determined by the parent dtype.
///
/// [`try_new`](Self::try_new) also enforces the invariants that make the split lossless:
///
/// - Each normalized row has L2 norm `1.0` or `0.0`, within the tolerance for its precision and
///   width.
/// - Each stored norm is non-negative.
/// - A stored norm is zero exactly when its normalized row is all zeros.
///
/// These checks include null rows. [`normalize`] zeroes both children at null positions.
///
/// # Lossy normalized children
///
/// Unchecked construction permits an approximate normalized child, such as a quantized direction.
/// If it can no longer prove the UnitVector tolerance, a Vector direction **must** erase that
/// refinement to ordinary Vector. The stored norms remain authoritative. [`L2Norm`],
/// [`InnerProduct`], and [`CosineSimilarity`] therefore operate on the stored children and can
/// differ slightly from decoding and recomputing.
///
/// [`Vector`]: crate::vector::Vector
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`normalize`]: crate::encodings::normalized::normalize
/// [`L2Norm`]: crate::scalar_fns::l2_norm::L2Norm
/// [`InnerProduct`]: crate::scalar_fns::inner_product::InnerProduct
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
#[derive(Clone, Debug)]
pub struct Normalized;

/// The slots of a [`NormalizedArray`].
#[array_slots(Normalized)]
pub struct NormalizedSlots {
    /// The non-nullable tensor-like direction of each row.
    #[slot(0)]
    pub normalized: ArrayRef,

    /// The non-nullable primitive L2 norm of each row.
    #[slot(1)]
    pub norms: ArrayRef,

    /// The optional non-nullable boolean validity mask.
    #[slot(2)]
    pub validity: Option<ArrayRef>,
}

pub(super) const DATA_CHILDREN: usize = NormalizedSlots::COUNT - 1;

impl Normalized {
    /// Builds an exact [`NormalizedArray`] after validating its semantic invariants.
    ///
    /// This is the constructor for exact norm splits. It scans both children, so it costs
    /// `O(len * list_size)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children are structurally incompatible, or if they violate any of
    /// the semantic invariants listed on [`Normalized`].
    pub fn try_new(
        mut normalized: ArrayRef,
        norms: ArrayRef,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<NormalizedArray> {
        let dtype = decoded_dtype_for_direction(normalized.dtype(), validity.nullability())?;
        if dtype
            .as_extension_opt()
            .is_some_and(|dtype| dtype.is::<Vector>())
            && !normalized
                .dtype()
                .as_extension_opt()
                .is_some_and(|dtype| dtype.is::<AnyUnitVector>())
        {
            let extension: ExtensionArray = normalized.execute(ctx)?;
            // SAFETY: The semantic scan below validates the direction before it is returned.
            normalized = unsafe { UnitVector::new_unchecked(extension.storage_array().clone())? };
        }
        if dtype
            .as_extension_opt()
            .is_some_and(|dtype| dtype.is::<Vector>())
        {
            vortex_ensure!(
                normalized
                    .dtype()
                    .as_extension_opt()
                    .is_some_and(|dtype| dtype.is::<AnyUnitVector>()),
                "exact Normalized vector direction must be a UnitVector, got {}",
                normalized.dtype(),
            );
        }
        let array = Array::try_from_parts(normalized_parts(dtype, normalized, norms, validity))?;
        validate_normalized_rows(array.normalized(), Some(array.norms()), ctx)?;

        Ok(array)
    }

    /// Builds a [`NormalizedArray`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must uphold the structural invariants listed on [`Normalized`]. In particular,
    /// both children must be non-nullable and have the same length. `normalized` must be a float
    /// tensor, and `norms` must be a primitive column with the same element ptype. `validity` must
    /// describe the same number of rows. An array-backed validity must be a non-nullable boolean
    /// column.
    ///
    /// This does not check the unit-norm relationship. Violating it can produce wrong answers but
    /// not memory unsafety.
    pub unsafe fn new_unchecked(
        normalized: ArrayRef,
        norms: ArrayRef,
        validity: Validity,
    ) -> NormalizedArray {
        let dtype = decoded_dtype_for_direction(normalized.dtype(), validity.nullability())
            .vortex_expect("new_unchecked requires a valid tensor direction dtype");
        unsafe { Self::new_unchecked_with_dtype(dtype, normalized, norms, validity) }
    }

    /// Builds a [`NormalizedArray`] with an explicit decoded dtype and without validation.
    ///
    /// # Safety
    ///
    /// The caller must uphold the invariants of [`new_unchecked`](Self::new_unchecked), and the
    /// normalized child must be compatible with `dtype`. A vector parent accepts a UnitVector
    /// child, or an ordinary Vector child when a documented lossy transform erased the
    /// refinement. Violating these requirements can produce incorrect results but not memory
    /// unsafety.
    pub(crate) unsafe fn new_unchecked_with_dtype(
        dtype: DType,
        normalized: ArrayRef,
        norms: ArrayRef,
        validity: Validity,
    ) -> NormalizedArray {
        unsafe { Array::from_parts_unchecked(normalized_parts(dtype, normalized, norms, validity)) }
    }
}

fn decoded_dtype_for_direction(
    direction_dtype: &DType,
    nullability: Nullability,
) -> VortexResult<DType> {
    if direction_dtype
        .as_extension_opt()
        .is_some_and(|dtype| dtype.is::<AnyVector>())
    {
        let storage_dtype = direction_dtype
            .as_extension()
            .storage_dtype()
            .with_nullability(nullability);
        let dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?;
        return Ok(DType::Extension(dtype.erased()));
    }

    Ok(direction_dtype.with_nullability(nullability))
}

fn normalized_parts(
    dtype: DType,
    normalized: ArrayRef,
    norms: ArrayRef,
    validity: Validity,
) -> ArrayParts<Normalized> {
    let len = normalized.len();
    let slots = NormalizedSlots {
        normalized,
        norms,
        validity: validity_to_child(&validity, len),
    }
    .into_slots();

    ArrayParts::new(Normalized, dtype, len, EmptyArrayData).with_slots(slots)
}

#[derive(Clone, prost::Message)]
struct NormalizedMetadata {
    /// The direction dtype, absent from legacy metadata where it matched the parent dtype.
    #[prost(message, optional, tag = "1")]
    direction_dtype: Option<pb::DType>,
}

impl VTable for Normalized {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.tensor.normalized");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = NormalizedSlotsView::from_slots(slots);

        validate_normalized_children(slots.normalized, slots.norms, slots.validity, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("NormalizedArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("NormalizedArray buffer_name index {idx} out of bounds")
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
        let direction_dtype = Some(array.slots_view().normalized.dtype().try_into()?);
        Ok(Some(NormalizedMetadata { direction_dtype }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let element_ptype = validate_tensor_float_input(dtype)?.element_ptype();
        let metadata = NormalizedMetadata::decode(metadata)
            .map_err(|error| vortex_err!("Failed to decode NormalizedMetadata: {error}"))?;
        let normalized_dtype = metadata
            .direction_dtype
            .as_ref()
            .map(|dtype| DType::from_proto(dtype, session))
            .transpose()?
            .unwrap_or_else(|| dtype.as_nonnullable());
        let norms_dtype = DType::Primitive(element_ptype, Nullability::NonNullable);

        let normalized = children.get(0, &normalized_dtype, len)?;
        let norms = children.get(1, &norms_dtype, len)?;

        // An absent validity child means "no nulls". The parent's nullability is what distinguishes
        // `NonNullable` from `AllValid`.
        let validity = match children.len() {
            DATA_CHILDREN => Validity::from(dtype.nullability()),
            NormalizedSlots::COUNT => {
                vortex_ensure!(
                    dtype.is_nullable(),
                    "Normalized validity child requires a nullable dtype, got {dtype}",
                );
                Validity::Array(children.get(NormalizedSlots::VALIDITY, &Validity::DTYPE, len)?)
            }
            other => vortex_bail!(
                "Normalized expects {DATA_CHILDREN} or {} children, got {other}",
                NormalizedSlots::COUNT,
            ),
        };

        Ok(normalized_parts(dtype.clone(), normalized, norms, validity))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        NormalizedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let dtype = array.dtype().clone();
        let validity = array.validity()?;
        let slots = array.slots_view();

        denormalize(slots.normalized, slots.norms, validity, dtype, ctx).map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl ValidityVTable<Normalized> for Normalized {
    fn validity(array: ArrayView<'_, Normalized>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.slots_view().validity,
            array.dtype().nullability(),
        ))
    }
}

impl OperationsVTable<Normalized> for Normalized {
    fn scalar_at(
        array: ArrayView<'_, Normalized>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Denormalize a single row rather than the whole column: both children are collapsed to
        // one-row constants, which also lets the constant-norms fast path do the multiply.
        let normalized = array.normalized().execute_scalar(index, ctx)?;
        let norms = array.norms().execute_scalar(index, ctx)?;
        let dtype = array.dtype().clone();

        // `Array::execute_scalar` resolves null rows before dispatching here, so this row is valid
        // and only the parent's nullability has to be reproduced.
        let validity = Validity::from(dtype.nullability());

        let row = denormalize(
            &ConstantArray::new(normalized, 1).into_array(),
            &ConstantArray::new(norms, 1).into_array(),
            validity,
            dtype,
            ctx,
        )?;

        row.execute_scalar(0, ctx)
    }
}
