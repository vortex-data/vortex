// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort the values array of a [`DictArray`] and remap codes accordingly.
//!
//! A "sorted" dictionary stores `values` in ascending order so that `codes` form an
//! order-preserving encoding of the original column. This unlocks O(1) min/max,
//! cheap is_sorted, and range-predicate pushdown into the codes domain.
//!
//! ## Performance notes
//!
//! The hot paths are:
//! 1. **argsort** of the dictionary values: O(d log d). The dict size `d` is bounded by
//!    `DictConstraints::max_len` (typically <= 64k), so this is dominated by comparison
//!    cost. For primitives we sort `(T, u32)` pairs to keep comparisons local in cache.
//! 2. **code remap**: O(n) over the codes array. The codes can be millions of entries, so
//!    we widen `inv_perm` once into a typed `Vec<C>` (matching the codes ptype) and do an
//!    unchecked gather loop.

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

/// Sort the values of an existing `DictArray` and remap its codes.
///
/// Preserves `all_values_referenced`. Sets `sorted_values = true` on the result.
pub fn sort_dict(dict: DictArray) -> VortexResult<DictArray> {
    let all_values_referenced = dict.has_all_values_referenced();

    if dict.has_sorted_values() {
        return Ok(dict);
    }

    let values = dict.values().clone();
    let codes = dict.codes().clone();

    if values.is_empty() {
        return Ok(unsafe {
            dict.set_sorted_values(true)
                .set_all_values_referenced(all_values_referenced)
        });
    }

    // perm[new_idx] = old_idx
    let perm = argsort_values(&values)?;

    let sorted_values = take_values(&values, &perm)?;
    let remapped_codes = remap_codes(&codes, &perm)?;

    Ok(unsafe {
        DictArray::new_unchecked(remapped_codes, sorted_values)
            .set_all_values_referenced(all_values_referenced)
            .set_sorted_values(true)
    })
}

/// Returns a permutation `perm` such that `values.take(perm)` is sorted ascending.
/// Nulls sort first.
fn argsort_values(values: &ArrayRef) -> VortexResult<Vec<u32>> {
    let n = u32::try_from(values.len())
        .map_err(|_| vortex_error::vortex_err!("dict values length {} exceeds u32::MAX", values.len()))?;

    if let Some(prim) = values.as_opt::<Primitive>() {
        let owned = prim.clone().into_owned();
        let mut perm = Vec::with_capacity(n as usize);
        match_each_native_ptype!(owned.ptype(), |P| {
            argsort_primitive::<P>(&owned, &mut perm)?;
        });
        Ok(perm)
    } else if let Some(vbv) = values.as_opt::<VarBinView>() {
        argsort_varbinview(&vbv.into_owned(), n)
    } else {
        // Canonicalize and retry.
        #[expect(deprecated)]
        match values.dtype() {
            crate::dtype::DType::Primitive(_, _) => {
                let prim = values.to_primitive();
                let mut perm = Vec::with_capacity(n as usize);
                match_each_native_ptype!(prim.ptype(), |P| {
                    argsort_primitive::<P>(&prim, &mut perm)?;
                });
                Ok(perm)
            }
            _ => {
                let vbv = values.to_varbinview();
                argsort_varbinview(&vbv, n)
            }
        }
    }
}

/// Argsort a primitive array. Builds `(value, idx)` pairs and sorts them in place;
/// this is faster than indirect comparison-based argsort because comparisons hit
/// adjacent memory rather than chasing pointers into the source slice.
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
            // Build pairs and sort by value. For Ord types we'd use sort_by_key, but Vortex's
            // NativePType uses total_compare (which handles NaN deterministically for floats).
            let mut pairs: Vec<(T, u32)> = slice
                .iter()
                .copied()
                .zip(0u32..)
                .map(|(v, i)| (v, i))
                .collect();
            pairs.sort_unstable_by(|a, b| a.0.total_compare(b.0));
            out_perm.extend(pairs.into_iter().map(|(_, i)| i));
        }
        Validity::AllInvalid => {
            // All-null: any permutation is sorted. Identity is cheapest.
            out_perm.extend(0u32..(n as u32));
        }
        Validity::Array(_) => {
            // Build (validity, value, idx) triples; validity=false sorts first.
            let valid: Vec<bool> = values.with_iterator(|it: &mut dyn Iterator<Item = Option<&T>>| {
                it.map(|opt| opt.is_some()).collect()
            });
            // We encode null-first by storing a u8 flag; pairs are small so this is fine.
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

/// Argsort a VarBinView array without copying the value bytes. We borrow byte slices
/// directly from the underlying data buffers via `bytes_at`-equivalent indexing.
fn argsort_varbinview(values: &VarBinViewArray, n: u32) -> VortexResult<Vec<u32>> {
    let views = values.views();
    let buffers: Vec<&[u8]> = (0..values.data_buffers().len())
        .map(|i| values.buffer(i).as_slice())
        .collect();
    let validity = values.validity()?;

    // Resolve each view to a (validity_flag, &[u8] slice). For inlined views, the slice
    // lives inside the view itself; we use a small workaround to expose it without copy.
    // We store the resolved slice as (start_ptr, len) pairs alongside an index.
    //
    // For sorting, we wrap into a `Vec<(u8, ResolvedView, u32)>` so null-first is cheap.

    enum Resolved<'a> {
        Inlined([u8; 12], u32), // value bytes + len (<=12)
        Ref(&'a [u8]),
    }

    impl<'a> Resolved<'a> {
        #[inline]
        fn as_slice(&self) -> &[u8] {
            match self {
                Resolved::Inlined(buf, len) => &buf[..*len as usize],
                Resolved::Ref(s) => s,
            }
        }
    }

    let valid_iter: Box<dyn Iterator<Item = bool>> = match &validity {
        Validity::NonNullable | Validity::AllValid => Box::new(std::iter::repeat_n(true, n as usize)),
        Validity::AllInvalid => Box::new(std::iter::repeat_n(false, n as usize)),
        Validity::Array(_) => {
            let v: Vec<bool> = values
                .with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
                    it.map(|opt| opt.is_some()).collect()
                });
            Box::new(v.into_iter())
        }
    };

    let mut triples: Vec<(u8, Resolved, u32)> = Vec::with_capacity(n as usize);
    for (idx, valid) in (0u32..).zip(valid_iter).take(n as usize) {
        let view = &views[idx as usize];
        let resolved = if view.is_inlined() {
            let inlined_bytes = view.as_inlined().value();
            let mut buf = [0u8; 12];
            let len = inlined_bytes.len().min(12);
            buf[..len].copy_from_slice(&inlined_bytes[..len]);
            Resolved::Inlined(buf, len as u32)
        } else {
            let r = view.as_view();
            let buf = buffers[r.buffer_index as usize];
            Resolved::Ref(&buf[r.as_range()])
        };
        triples.push((u8::from(valid), resolved, idx));
    }

    triples.sort_unstable_by(|a, b| match a.0.cmp(&b.0) {
        Ordering::Equal => a.1.as_slice().cmp(b.1.as_slice()),
        ord => ord,
    });

    Ok(triples.into_iter().map(|(_, _, i)| i).collect())
}

/// Reorder `values` such that the i-th element of the output is `values[perm[i]]`.
fn take_values(values: &ArrayRef, perm: &[u32]) -> VortexResult<ArrayRef> {
    let indices = BufferMut::<u32>::from_iter(perm.iter().copied())
        .freeze()
        .into_array();
    values.take(indices)
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
