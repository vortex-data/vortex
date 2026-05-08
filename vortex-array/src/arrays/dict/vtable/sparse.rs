// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::super::array::DictSlots;
use super::super::array::compute_referenced_values_mask_from_codes;
use super::DictArray;
use super::cardinality;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;

// TODO: Replace this fixed sparse-dictionary threshold with a cost model that accounts for values
// encoding, code count, unique-code count, and exporter/canonicalization costs.
const SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD: usize = 4;
const SPARSE_CANONICALIZE_SAMPLED_CODES_PER_VALUE_THRESHOLD: usize = 2;
const SPARSE_CANONICALIZE_MIN_SAMPLED_VALUES_LEN: usize = 512;

struct SparseDictCodes {
    /// Original dictionary value indices that are actually referenced by the live codes.
    unique_codes: PrimitiveArray,
    /// Codes rewritten to index into `unique_codes` instead of the original values array.
    remapped_codes: PrimitiveArray,
}

#[cold]
#[inline(never)]
pub(super) fn sparse_canonicalize_dict(
    array: &DictArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Canonical>> {
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

#[cold]
#[inline(never)]
fn collect_sparse_codes(
    codes: &PrimitiveArray,
    values_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<SparseDictCodes>> {
    let validity = codes.validity()?;
    let validity_mask = validity.execute_mask(codes.len(), ctx)?;
    let codes = codes_as_u64(codes, ctx)?;

    // The exact pass below scans every code and allocates a remap table sized to the values array.
    // Do it only when a cheap upper bound/sample says the dictionary is likely sparse enough.
    if !should_collect_sparse_codes(&codes, values_len, &validity_mask) {
        return Ok(None);
    }

    let referenced_values =
        compute_referenced_values_mask_from_codes(&codes, values_len, &validity_mask, true)?;
    let unique_count = referenced_values.true_count();
    if unique_count.saturating_mul(SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD) >= values_len {
        return Ok(None);
    }

    collect_sparse_codes_u64(&codes, referenced_values, validity_mask, validity).map(Some)
}

fn codes_as_u64(codes: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    if codes.ptype() == PType::U64 {
        return Ok(codes.clone());
    }

    codes
        .clone()
        .into_array()
        .cast(DType::Primitive(PType::U64, codes.dtype().nullability()))?
        .execute::<PrimitiveArray>(ctx)
}

#[cold]
#[inline(never)]
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

    if !should_sample_sparse_canonicalize(codes.len(), values_len) {
        return false;
    }

    // Otherwise sample first. This catches cases like many live rows all referencing the same
    // dictionary value without forcing dense dictionaries through the exact remap scan.
    if !cardinality::has_repeated_code_sample(codes, validity_mask) {
        return false;
    }

    let Some(estimated_unique_codes) = cardinality::estimate_code_cardinality(codes, validity_mask)
    else {
        return false;
    };

    estimated_unique_codes.saturating_mul(SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD)
        < values_len
}

#[inline]
pub(super) fn should_consider_sparse_canonicalize(codes_len: usize, values_len: usize) -> bool {
    codes_len.saturating_mul(SPARSE_CANONICALIZE_CODES_PER_VALUE_THRESHOLD) < values_len
        || should_sample_sparse_canonicalize(codes_len, values_len)
}

#[inline]
fn should_sample_sparse_canonicalize(codes_len: usize, values_len: usize) -> bool {
    // Sampling is only a preflight for cases that are not sparse by row count alone. Keep it away
    // from tiny dictionary domains and near-dense slices where the estimator overhead dominates.
    values_len >= SPARSE_CANONICALIZE_MIN_SAMPLED_VALUES_LEN
        && codes_len.saturating_mul(SPARSE_CANONICALIZE_SAMPLED_CODES_PER_VALUE_THRESHOLD)
            < values_len
}

#[cold]
#[inline(never)]
fn collect_sparse_codes_u64(
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
    for (idx, &code) in codes.as_slice::<u64>().iter().enumerate() {
        if !validity_mask.value(idx) {
            remapped_codes.push(0);
            continue;
        }

        let old_code = usize::try_from(code)
            .unwrap_or_else(|_| vortex_panic!("dictionary code {code} does not fit usize"));
        let new_code = value_remap[old_code];
        debug_assert_ne!(new_code, usize::MAX);

        remapped_codes.push(new_code as u64);
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
            PrimitiveArray::from_option_iter([Some(0u64), None, Some(1), Some(0)]).into_array()
        );

        Ok(())
    }

    #[test]
    fn sampled_sparse_codes_remaps_repeated_large_codes() -> VortexResult<()> {
        let codes = PrimitiveArray::from_iter((0..1024).map(|_| 42u32));
        let Some(sparse) =
            collect_sparse_codes(&codes, 3000, &mut LEGACY_SESSION.create_execution_ctx())?
        else {
            panic!("sampled codes are sparse");
        };

        assert_arrays_eq!(
            sparse.unique_codes.into_array(),
            PrimitiveArray::from_iter([42u64]).into_array()
        );
        assert_arrays_eq!(
            sparse.remapped_codes.into_array(),
            PrimitiveArray::from_iter((0..1024).map(|_| 0u64)).into_array()
        );

        Ok(())
    }

    #[test]
    fn dense_sample_skips_sparse_code_collection() -> VortexResult<()> {
        let codes = PrimitiveArray::from_iter((0..1024).map(|idx| idx as u32));

        assert!(
            collect_sparse_codes(&codes, 3000, &mut LEGACY_SESSION.create_execution_ctx())?
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
            PrimitiveArray::from_iter(0..3000i32).into_array(),
        );

        let actual = dict
            .into_array()
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
            .into_array();

        assert_arrays_eq!(actual, PrimitiveArray::from_iter((0..1024).map(|_| 42i32)));

        Ok(())
    }
}
