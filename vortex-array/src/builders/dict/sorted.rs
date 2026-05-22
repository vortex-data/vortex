// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort the values array of a [`DictArray`] and remap codes accordingly so the codes
//! form an order-preserving encoding of the original column. This unlocks O(1) min/max,
//! cheap is_sorted, and range-predicate pushdown into the codes domain.

use std::cmp::Ordering;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::DictConstraints;
use super::dict_encode_with_constraints;
use crate::ArrayRef;
use crate::IntoArray;
#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::accessor::ArrayAccessor;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

/// Encode `array` as a dictionary whose values are sorted in ascending order.
///
/// Nulls sort first.
pub fn dict_encode_sorted(array: &ArrayRef) -> VortexResult<DictArray> {
    let dict = dict_encode_with_constraints(array, &super::UNCONSTRAINED)?;
    if dict.len() != array.len() {
        vortex_bail!(
            "dict_encode_sorted must have encoded all {} elements, but only encoded {}",
            array.len(),
            dict.len(),
        );
    }
    sort_dict(dict)
}

/// Encode `array` as a dictionary subject to constraints, with sorted values.
pub fn dict_encode_sorted_with_constraints(
    array: &ArrayRef,
    constraints: &DictConstraints,
) -> VortexResult<DictArray> {
    let dict = dict_encode_with_constraints(array, constraints)?;
    sort_dict(dict)
}

/// Sort the values of an existing `DictArray` and remap its codes. The resulting values
/// array has `Stat::IsSorted = Exact(true)` cached so `dict.has_sorted_values()` returns
/// true without any further work.
pub fn sort_dict(dict: DictArray) -> VortexResult<DictArray> {
    let all_values_referenced = dict.has_all_values_referenced();

    if dict.has_sorted_values() {
        return Ok(dict);
    }

    let values = dict.values().clone();
    let codes = dict.codes().clone();

    if values.is_empty() {
        return Ok(unsafe { dict.set_all_values_referenced(all_values_referenced) });
    }

    // perm[new_idx] = old_idx
    let perm = argsort_values(&values)?;

    let sorted_values = take_values(&values, &perm)?;
    let remapped_codes = remap_codes(&codes, &perm)?;

    // Stat is the single source of truth for sortedness — set it on the values so the
    // dict's `has_sorted_values()` picks it up. The stat persists through the layout
    // writer / reader and through transforms that share the values Arc.
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    sorted_values
        .statistics()
        .set(Stat::IsSorted, Precision::Exact(true.into()));

    Ok(unsafe {
        DictArray::new_unchecked(remapped_codes, sorted_values)
            .set_all_values_referenced(all_values_referenced)
    })
}

/// Returns a permutation `perm` such that `values.take(perm)` is sorted ascending.
/// Nulls sort first.
fn argsort_values(values: &ArrayRef) -> VortexResult<Vec<u32>> {
    let n = u32::try_from(values.len())
        .map_err(|_| vortex_error::vortex_err!("dict values length {} exceeds u32::MAX", values.len()))?;

    // Resolve to a canonical Primitive or VarBinView, canonicalizing only if needed.
    if let Some(prim) = values.as_opt::<Primitive>() {
        argsort_primitive_array(&prim.into_owned())
    } else if let Some(vbv) = values.as_opt::<VarBinView>() {
        argsort_varbinview(&vbv.into_owned(), n)
    } else if values.dtype().is_primitive() {
        #[expect(deprecated)]
        argsort_primitive_array(&values.to_primitive())
    } else {
        #[expect(deprecated)]
        argsort_varbinview(&values.to_varbinview(), n)
    }
}

fn argsort_primitive_array(prim: &PrimitiveArray) -> VortexResult<Vec<u32>> {
    let mut perm = Vec::with_capacity(prim.len());
    match_each_native_ptype!(prim.ptype(), |P| {
        argsort_primitive::<P>(prim, &mut perm)?;
    });
    Ok(perm)
}

/// Argsort a primitive array. Sorts `(value, idx)` pairs to keep comparisons in cache.
fn argsort_primitive<T: NativePType>(
    values: &PrimitiveArray,
    out_perm: &mut Vec<u32>,
) -> VortexResult<()> {
    let slice = values.as_slice::<T>();
    let n = slice.len();
    out_perm.clear();
    out_perm.reserve(n);

    match values.validity()? {
        Validity::NonNullable | Validity::AllValid => {
            // total_compare handles NaN deterministically; sort_unstable_by_key needs Ord
            // which floats lack.
            let mut pairs: Vec<(T, u32)> = slice.iter().copied().zip(0u32..).collect();
            pairs.sort_unstable_by(|a, b| a.0.total_compare(b.0));
            out_perm.extend(pairs.into_iter().map(|(_, i)| i));
        }
        Validity::AllInvalid => {
            // All-null: any permutation is sorted; identity is cheapest.
            let n_u32 = u32::try_from(n).map_err(|_| {
                vortex_error::vortex_err!("dict values length {n} exceeds u32::MAX")
            })?;
            out_perm.extend(0u32..n_u32);
        }
        Validity::Array(_) => {
            // Sort (validity, value, idx) — null (validity=0) sorts before non-null.
            let valid: Vec<bool> = values.with_iterator(|it: &mut dyn Iterator<Item = Option<&T>>| {
                it.map(|opt| opt.is_some()).collect()
            });
            let mut triples: Vec<(u8, T, u32)> = slice
                .iter()
                .copied()
                .zip(0u32..)
                .map(|(v, i)| (u8::from(valid[i as usize]), v, i))
                .collect();
            triples.sort_unstable_by(|a, b| match a.0.cmp(&b.0) {
                Ordering::Equal => a.1.total_compare(b.1),
                ord => ord,
            });
            out_perm.extend(triples.into_iter().map(|(_, _, i)| i));
        }
    }
    Ok(())
}

/// Argsort a VarBinView. Materializes each value to `Vec<u8>` once via the standard
/// `ArrayAccessor::with_iterator` (dict_len <= u16::MAX so this is cheap), then sorts a
/// `(validity, bytes, idx)` triple by validity-first then bytes.
fn argsort_varbinview(values: &VarBinViewArray, n: u32) -> VortexResult<Vec<u32>> {
    let items: Vec<Option<Vec<u8>>> = values
        .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
            it.map(|opt| opt.map(|b| b.to_vec())).collect()
        });
    let mut perm: Vec<u32> = (0..n).collect();
    perm.sort_unstable_by(|&a, &b| {
        match (&items[a as usize], &items[b as usize]) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Less, // nulls sort first
            (Some(_), None) => Ordering::Greater,
            (Some(av), Some(bv)) => av.as_slice().cmp(bv.as_slice()),
        }
    });
    Ok(perm)
}

/// Reorder `values` such that the i-th element of the output is `values[perm[i]]`.
///
/// `Array::take` wraps in a `DictArray` and optimizes lazily; the downstream sorted-aware
/// kernels expect the canonical underlying values array (Primitive / VarBinView) for the
/// typed linear scan, so we eagerly canonicalize here.
fn take_values(values: &ArrayRef, perm: &[u32]) -> VortexResult<ArrayRef> {
    use crate::Canonical;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;

    let indices = PrimitiveArray::from_iter(perm.iter().copied()).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    values
        .take(indices)?
        .execute::<Canonical>(&mut ctx)
        .map(Into::into)
}

/// Apply the sort permutation to a codes array: for each code `c`, replace with
/// `inv_perm[c]` where `inv_perm` is the inverse of `perm`.
///
/// Optimized path:
/// 1. Build `inv_perm` typed to the codes ptype (one allocation, no per-element conversion).
/// 2. Gather into the output with a tight unchecked loop.
fn remap_codes(codes: &ArrayRef, perm: &[u32]) -> VortexResult<ArrayRef> {
    #[expect(deprecated)]
    let codes = codes.to_primitive();
    let ptype = codes.ptype();
    let validity = codes.validity()?;
    let result = match_each_unsigned_integer_ptype!(ptype, |P| {
        remap_codes_typed::<P>(&codes, perm, validity)?
    });
    Ok(result)
}

fn remap_codes_typed<C: UnsignedPType>(
    codes: &PrimitiveArray,
    perm: &[u32],
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let slice = codes.as_slice::<C>();
    let dict_len = perm.len();

    // Sanity: dict_len fits in the codes ptype (it did before the sort, and the sort
    // preserves dict_len).
    let max_code = C::PTYPE.max_value_as_u64();
    if dict_len as u64 > max_code.saturating_add(1) {
        vortex_error::vortex_bail!(
            "dict length {} exceeds maximum code value for ptype {}",
            dict_len,
            C::PTYPE
        );
    }

    // Build typed inverse permutation in one pass: inv[perm[i]] = i.
    let mut inv: Vec<C> = vec![C::default(); dict_len];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        // SAFETY: perm holds a permutation of 0..dict_len so old_idx < dict_len; new_idx
        // also fits in C because dict_len fits in C (checked above).
        let new_code = C::from_usize(new_idx).vortex_expect("new_idx fits in C");
        unsafe {
            *inv.get_unchecked_mut(old_idx as usize) = new_code;
        }
    }

    let n = slice.len();
    let mut out = BufferMut::<C>::with_capacity(n);
    // SAFETY: we reserved exactly `n` slots and write each one below.
    unsafe {
        let dst = out.spare_capacity_mut().as_mut_ptr().cast::<C>();
        for i in 0..n {
            // SAFETY: codes in slice are valid indices < dict_len by DictArray invariant.
            let c = *slice.get_unchecked(i);
            let new_code = *inv.get_unchecked(c.as_());
            std::ptr::write(dst.add(i), new_code);
        }
        out.set_len(n);
    }

    Ok(PrimitiveArray::new(out, validity).into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::dict::dict_encode;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[test]
    fn sort_primitive_dict() {
        let arr = buffer![3i32, 1, 2, 1, 3, 2].into_array();
        let sorted = dict_encode_sorted(&arr).unwrap();
        assert!(sorted.has_sorted_values());
        assert!(sorted.has_all_values_referenced());

        let expected_values = buffer![1i32, 2, 3].into_array();
        assert_arrays_eq!(sorted.values(), expected_values);

        let expected_codes = buffer![2u8, 0, 1, 0, 2, 1].into_array();
        assert_arrays_eq!(sorted.codes(), expected_codes);
    }

    #[test]
    fn sort_primitive_dict_with_nulls() {
        let arr = PrimitiveArray::from_option_iter([
            Some(5i32),
            None,
            Some(1),
            Some(5),
            None,
            Some(3),
        ])
        .into_array();
        let sorted = dict_encode_sorted(&arr).unwrap();
        assert!(sorted.has_sorted_values());

        #[expect(deprecated)]
        let canon = sorted.as_array().to_primitive();
        assert_arrays_eq!(canon, PrimitiveArray::from_option_iter([
            Some(5i32),
            None,
            Some(1),
            Some(5),
            None,
            Some(3),
        ]));
    }

    #[test]
    fn sort_varbin_dict() {
        let arr = VarBinArray::from(vec!["zeta", "alpha", "mu", "alpha", "zeta"]).into_array();
        let sorted = dict_encode_sorted(&arr).unwrap();
        assert!(sorted.has_sorted_values());

        #[expect(deprecated)]
        let canon = sorted.as_array().to_varbinview();
        canon.with_iterator(|it| {
            let strs: Vec<_> = it
                .map(|b| b.map(|s| std::str::from_utf8(s).unwrap().to_string()))
                .collect();
            assert_eq!(
                strs,
                vec![
                    Some("zeta".to_string()),
                    Some("alpha".to_string()),
                    Some("mu".to_string()),
                    Some("alpha".to_string()),
                    Some("zeta".to_string()),
                ]
            );
        });
    }

    #[test]
    fn sort_dict_idempotent() {
        let arr = buffer![3i32, 1, 2].into_array();
        let once = dict_encode_sorted(&arr).unwrap();
        let twice = sort_dict(once.clone()).unwrap();
        assert!(twice.has_sorted_values());
        assert_arrays_eq!(once.values(), twice.values());
    }

    #[test]
    fn sort_dict_preserves_all_values_referenced() {
        let arr = buffer![1i32, 2, 3, 2, 1].into_array();
        let dict = dict_encode(&arr).unwrap();
        assert!(dict.has_all_values_referenced());
        let sorted = sort_dict(dict).unwrap();
        assert!(sorted.has_sorted_values());
        assert!(sorted.has_all_values_referenced());
    }

    #[test]
    fn sort_empty_dict() {
        let values = buffer![1i32, 2, 3].into_array();
        let codes = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        let dict = DictArray::try_new(codes, values).unwrap();
        let sorted = sort_dict(dict).unwrap();
        assert!(sorted.has_sorted_values());
    }

    #[test]
    fn sort_nullable_dtype() {
        let arr = VarBinArray::from_iter(
            [Some("z"), None, Some("a"), Some("z"), None, Some("m")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let sorted = dict_encode_sorted(&arr).unwrap();
        assert!(sorted.has_sorted_values());
    }

    #[test]
    fn sort_long_string_values() {
        // Force the non-inlined path: strings > 12 bytes.
        let arr = VarBinArray::from(vec![
            "this_is_a_long_zeta",
            "this_is_a_long_alpha",
            "this_is_a_long_mu",
            "this_is_a_long_alpha",
            "this_is_a_long_zeta",
        ])
        .into_array();
        let sorted = dict_encode_sorted(&arr).unwrap();
        assert!(sorted.has_sorted_values());

        #[expect(deprecated)]
        let canon = sorted.as_array().to_varbinview();
        canon.with_iterator(|it| {
            let strs: Vec<_> = it
                .map(|b| b.map(|s| std::str::from_utf8(s).unwrap().to_string()))
                .collect();
            assert_eq!(
                strs,
                vec![
                    Some("this_is_a_long_zeta".to_string()),
                    Some("this_is_a_long_alpha".to_string()),
                    Some("this_is_a_long_mu".to_string()),
                    Some("this_is_a_long_alpha".to_string()),
                    Some("this_is_a_long_zeta".to_string()),
                ]
            );
        });
    }

    #[test]
    fn sort_large_dict() {
        // Verify correctness at a slightly larger scale (forces u16 codes).
        let mut data: Vec<i32> = (0..1000).rev().collect();
        for _ in 0..3 {
            data.extend((0..1000).rev());
        }
        let arr = PrimitiveArray::from_iter(data.clone()).into_array();
        let sorted = dict_encode_sorted(&arr).unwrap();
        assert!(sorted.has_sorted_values());

        // Values should be 0..1000.
        let expected_values = PrimitiveArray::from_iter(0i32..1000).into_array();
        assert_arrays_eq!(sorted.values(), expected_values);

        // Decoded should equal original.
        #[expect(deprecated)]
        let canon = sorted.as_array().to_primitive();
        assert_arrays_eq!(canon, PrimitiveArray::from_iter(data));
    }
}
