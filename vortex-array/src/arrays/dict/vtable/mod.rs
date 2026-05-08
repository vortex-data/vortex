// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use kernel::PARENT_KERNELS;
use num_traits::FromPrimitive;
use prost::Message;
use smallvec::smallvec;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DictData;
use super::DictMetadata;
use super::DictOwnedExt;
use super::DictParts;
use super::array::DictSlots;
use super::array::DictSlotsView;
use super::array::compute_referenced_values_mask_from_codes;
use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::compute::rules::PARENT_RULES;
use crate::arrays::dict::execute::take_canonical;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_integer_ptype;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

mod cardinality;
mod kernel;
mod operations;
mod validity;

/// A [`Dict`]-encoded Vortex array.
pub type DictArray = Array<Dict>;

// TODO: Replace this fixed sparse-dictionary threshold with a cost model that accounts for values
// encoding, code count, unique-code count, and exporter/canonicalization costs.
const SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD: usize = 4;

#[derive(Clone, Debug)]
pub struct Dict;

impl ArrayHash for DictData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for DictData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

impl VTable for Dict {
    type TypedArrayData = DictData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.dict");
        *ID
    }

    fn validate(
        &self,
        _data: &DictData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let view = DictSlotsView::from_slots(slots);
        let codes = view.codes;
        let values = view.values;
        vortex_ensure!(codes.len() == len, "DictArray codes length mismatch");
        vortex_ensure!(
            values
                .dtype()
                .union_nullability(codes.dtype().nullability())
                == *dtype,
            "DictArray dtype does not match codes/values dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DictArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DictMetadata {
                codes_ptype: PType::try_from(array.codes().dtype())? as i32,
                values_len: u32::try_from(array.values().len()).map_err(|_| {
                    vortex_err!(
                        "Dictionary values size {} overflowed u32",
                        array.values().len()
                    )
                })?,
                is_nullable_codes: Some(array.codes().dtype().is_nullable()),
                all_values_referenced: Some(array.has_all_values_referenced()),
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
        let metadata = DictMetadata::decode(metadata)?;
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;
        let all_values_referenced = metadata.all_values_referenced.unwrap_or(false);

        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, unsafe {
            DictData::new_unchecked().set_all_values_referenced(all_values_referenced)
        })
        .with_slots(smallvec![Some(codes), Some(values)]))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DictSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if array.is_empty() {
            let result_dtype = array
                .dtype()
                .union_nullability(array.codes().dtype().nullability());
            return Ok(ExecutionResult::done(Canonical::empty(&result_dtype)));
        }

        let array = require_child!(array, array.codes(), DictSlots::CODES => Primitive);

        if matches!(array.codes().validity()?, Validity::AllInvalid) {
            return Ok(ExecutionResult::done(ConstantArray::new(
                Scalar::null(array.dtype().as_nullable()),
                array.codes().len(),
            )));
        }

        if let Some(canonical) = sparse_canonicalize_dict(&array, ctx)? {
            return Ok(ExecutionResult::done(canonical));
        }

        let array = require_child!(array, array.values(), DictSlots::VALUES => AnyCanonical);

        let DictParts { values, codes, .. } = array.into_parts();

        Ok(ExecutionResult::done(take_canonical(
            values.as_::<AnyCanonical>(),
            &codes.downcast::<Primitive>(),
            ctx,
        )?))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

struct SparseDictCodes {
    /// Original dictionary value indices that are actually referenced by the live codes.
    unique_codes: PrimitiveArray,
    /// Codes rewritten to index into `unique_codes` instead of the original values array.
    remapped_codes: PrimitiveArray,
}

fn sparse_canonicalize_dict(
    array: &DictArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Canonical>> {
    // If metadata tells us every dictionary value is referenced, there is no garbage to compact.
    // This also keeps hot paths such as dictionary comparisons from paying the sparse estimator
    // cost when they produce dense, all-referenced result dictionaries.
    if array.has_all_values_referenced() {
        return Ok(None);
    }

    let codes = array.codes().as_::<Primitive>().into_owned();
    let Some(sparse_codes) = collect_sparse_codes(&codes, array.values().len(), ctx)? else {
        return Ok(None);
    };

    // Build a temporary parent that represents `values.take(unique_codes)`. Calling
    // `execute_parent` on the values child lets encodings such as FSST/VarBin sparse-take just
    // the referenced dictionary values. If the child has no specialized parent execution, fall
    // back to canonicalizing all values and then taking from the canonical array.
    let values = array.values();
    let unique_values_parent = DictArray::new(
        sparse_codes.unique_codes.clone().into_array(),
        values.clone(),
    )
    .into_array();
    let unique_values = if let Some(taken_values) =
        values.execute_parent(&unique_values_parent, DictSlots::VALUES, ctx)?
    {
        taken_values.execute::<Canonical>(ctx)?.into_array()
    } else {
        let canonical_values = values.clone().execute::<Canonical>(ctx)?.into_array();
        DictArray::new(sparse_codes.unique_codes.into_array(), canonical_values)
            .into_array()
            .execute::<Canonical>(ctx)?
            .into_array()
    };

    // Now the dictionary is dense over its compacted values, so normal dictionary execution only
    // takes from the small `unique_values` array. This avoids `values.take(codes)` preserving a
    // large dictionary with many unused values.
    let compact_dict = unsafe {
        DictArray::new_unchecked(sparse_codes.remapped_codes.into_array(), unique_values)
            .set_all_values_referenced(true)
    };

    compact_dict
        .into_array()
        .execute::<Canonical>(ctx)
        .map(Some)
}

fn collect_sparse_codes(
    codes: &PrimitiveArray,
    values_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<SparseDictCodes>> {
    let validity = codes.validity()?;
    let validity_mask = validity.execute_mask(codes.len(), ctx)?;

    // The exact pass below scans every code and allocates a remap table sized to the values array.
    // Do it only when a cheap upper bound/sample says the dictionary is likely sparse enough.
    if !should_collect_sparse_codes(codes, values_len, &validity_mask) {
        return Ok(None);
    }

    let referenced_values =
        compute_referenced_values_mask_from_codes(codes, values_len, &validity_mask, true)?;
    let unique_count = referenced_values.true_count();
    if unique_count.saturating_mul(SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD) >= values_len {
        return Ok(None);
    }

    let sparse_codes = match_each_integer_ptype!(codes.ptype(), |P| {
        collect_sparse_codes_typed::<P>(codes, referenced_values, validity_mask, validity)?
    });

    Ok(Some(sparse_codes))
}

fn should_collect_sparse_codes(
    codes: &PrimitiveArray,
    values_len: usize,
    validity_mask: &Mask,
) -> bool {
    if codes.is_empty() || values_len == 0 || validity_mask.true_count() == 0 {
        return false;
    }

    // If even the worst case "every live code is unique" is sparse, skip sampling and go straight
    // to the exact remap pass.
    if codes
        .len()
        .saturating_mul(SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD)
        < values_len
    {
        return true;
    }

    // Otherwise sample first. This catches cases like many live rows all referencing the same
    // dictionary value without forcing dense dictionaries through the exact remap scan.
    let Some(estimated_unique_codes) = match_each_integer_ptype!(codes.ptype(), |P| {
        cardinality::estimate_code_cardinality::<P>(codes, validity_mask)
    }) else {
        return false;
    };

    estimated_unique_codes.saturating_mul(SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD)
        < values_len
}

fn collect_sparse_codes_typed<P: IntegerPType + FromPrimitive>(
    codes: &PrimitiveArray,
    referenced_values: BitBuffer,
    validity_mask: Mask,
    validity: Validity,
) -> VortexResult<SparseDictCodes> {
    let unique_count = referenced_values.true_count();
    let mut value_remap = vec![usize::MAX; referenced_values.len()];
    let mut unique_codes = Vec::with_capacity(unique_count);

    // Reuse the same exact referenced-values bitmap as the dictionary aggregate kernels. Walking
    // the bitmap assigns compact codes in original dictionary order, which keeps compaction
    // deterministic and independent of the first live row that happened to reference each value.
    for old_code in referenced_values.set_indices() {
        let new_code = unique_codes.len();
        value_remap[old_code] = new_code;
        unique_codes.push(old_code as u64);
    }

    let mut remapped_codes = Vec::with_capacity(codes.len());
    for (idx, &code) in codes.as_slice::<P>().iter().enumerate() {
        if !validity_mask.value(idx) {
            remapped_codes.push(P::default());
            continue;
        }

        let old_code = code.as_();
        let new_code = value_remap[old_code];
        debug_assert_ne!(new_code, usize::MAX);

        remapped_codes.push(P::from_usize(new_code).unwrap_or_else(|| {
            vortex_panic!(
                "compacted dictionary code {new_code} does not fit in {}",
                P::PTYPE
            )
        }));
    }

    Ok(SparseDictCodes {
        unique_codes: PrimitiveArray::new(Buffer::from_iter(unique_codes), Validity::NonNullable),
        remapped_codes: PrimitiveArray::new(Buffer::from_iter(remapped_codes), validity),
    })
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;

    #[test]
    fn collect_sparse_codes_remaps_unique_values() -> VortexResult<()> {
        let codes = PrimitiveArray::from_option_iter([Some(50u32), None, Some(70), Some(50)]);
        let Some(sparse) =
            collect_sparse_codes(&codes, 100, &mut LEGACY_SESSION.create_execution_ctx())?
        else {
            panic!("codes are sparse");
        };

        assert_arrays_eq!(
            sparse.unique_codes.into_array(),
            PrimitiveArray::from_iter([50u64, 70]).into_array()
        );
        assert_arrays_eq!(
            sparse.remapped_codes.into_array(),
            PrimitiveArray::from_option_iter([Some(0u32), None, Some(1), Some(0)]).into_array()
        );

        Ok(())
    }

    #[test]
    fn sampled_sparse_codes_remaps_repeated_large_codes() -> VortexResult<()> {
        let codes = PrimitiveArray::from_iter((0..1024).map(|_| 42u32));
        let Some(sparse) =
            collect_sparse_codes(&codes, 100, &mut LEGACY_SESSION.create_execution_ctx())?
        else {
            panic!("sampled codes are sparse");
        };

        assert_arrays_eq!(
            sparse.unique_codes.into_array(),
            PrimitiveArray::from_iter([42u64]).into_array()
        );
        assert_arrays_eq!(
            sparse.remapped_codes.into_array(),
            PrimitiveArray::from_iter((0..1024).map(|_| 0u32)).into_array()
        );

        Ok(())
    }

    #[test]
    fn dense_sample_skips_sparse_code_collection() -> VortexResult<()> {
        let codes = PrimitiveArray::from_iter((0..1024).map(|idx| (idx % 100) as u32));

        assert!(
            collect_sparse_codes(&codes, 100, &mut LEGACY_SESSION.create_execution_ctx())?
                .is_none()
        );

        Ok(())
    }

    #[test]
    fn sparse_dict_canonicalizes_correctly() -> VortexResult<()> {
        let dict = DictArray::new(
            PrimitiveArray::from_option_iter([Some(50u32), None, Some(70), Some(50)]).into_array(),
            PrimitiveArray::from_iter(0..100i32).into_array(),
        );

        let actual = dict
            .into_array()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
            .into_array();

        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_option_iter([Some(50i32), None, Some(70), Some(50)])
        );

        Ok(())
    }

    #[test]
    fn sampled_sparse_dict_canonicalizes_repeated_codes() -> VortexResult<()> {
        let dict = DictArray::new(
            PrimitiveArray::from_iter((0..1024).map(|_| 42u32)).into_array(),
            PrimitiveArray::from_iter(0..100i32).into_array(),
        );

        let actual = dict
            .into_array()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
            .into_array();

        assert_arrays_eq!(actual, PrimitiveArray::from_iter((0..1024).map(|_| 42i32)));

        Ok(())
    }
}
