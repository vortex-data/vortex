// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused `list_contains` (`IN (...)`) kernel for [`BitPackedArray`].
//!
//! Partitions the non-null constant set members into in-packed-domain (`[0, 2^W - 1]`) and
//! patch-only (`< 0` or `>= 2^W`). When zero members fall in the packed domain, packed blocks are
//! never unpacked. When 1–4 distinct members fall in the packed domain, membership is evaluated
//! during FastLanes unpacking via [`stream_compare_fused_bits`]. Larger in-domain sets return
//! `Ok(None)` to fall back to [`PreparedSet`].
//!
//! [`BitPackedArray`]: crate::BitPackedArray

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Constant;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_functions::list_contains::ListContainsElementKernel;
use vortex_functions::list_contains::ListContainsOptions;
use vortex_functions::list_contains::PreparedSet;
use vortex_functions::list_contains::PreparedSetData;

use super::compare_fused::stream_compare_fused_bits;
use super::packed_range::constant_bits;
use super::packed_range::packed_range;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::unpack_iter::BitPacked as BitPackedIter;

const MAX_FUSED_MEMBERS: usize = 4;
/// Bail out before materializing `ListScalar::elements()` on large `IN` sets so chunked scans
/// reuse the shared `PreparedSet` haystack instead of paying O(|set|) work per chunk.
const MAX_SET_LEN: usize = 64;
const PATCH_BINARY_SEARCH_THRESHOLD: usize = 8;

impl ListContainsElementKernel for BitPacked {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = element.bit_widths() else {
            return Ok(None);
        };

        let prepared = PreparedSet::set_of(list);
        let Some(list_scalar) = prepared.map(PreparedSetData::list).or_else(|| {
            list.as_opt::<Constant>()
                .map(|constant| constant.data().scalar())
        }) else {
            return Ok(None);
        };

        let DType::List(member_dtype, _) = list_scalar.dtype() else {
            return Ok(None);
        };
        if !member_dtype.eq_ignore_nullability(element.dtype()) {
            return Ok(None);
        }
        let list_value = list_scalar.as_list();
        if list_value.len() > MAX_SET_LEN {
            return Ok(None);
        }
        let Some(list_elements) = list_value.elements() else {
            return Ok(None);
        };

        match_each_integer_ptype!(element.dtype().as_ptype(), |T| {
            list_contains_typed::<T>(
                prepared,
                list_scalar.dtype(),
                &list_elements,
                element,
                bit_width,
                options,
                ctx,
            )
        })
    }
}

fn list_contains_typed<T>(
    prepared: Option<&PreparedSetData>,
    list_dtype: &DType,
    list_elements: &[Scalar],
    element: ArrayView<'_, BitPacked>,
    bit_width: u8,
    options: &ListContainsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: IntegerPType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>
        + AsPrimitive<<T as PhysicalPType>::Physical>
        + TryFrom<PValue, Error = VortexError>,
    <T as PhysicalPType>::Physical: BitPacking
        + NativePType
        + BitPackingCompare
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
{
    let range = packed_range::<T>(bit_width);
    let has_patches = element.patches().is_some();

    let mut in_domain = [T::zero(); MAX_FUSED_MEMBERS];
    let mut in_domain_len = 0usize;
    let mut patch_only = Vec::new();
    let mut has_null_element = false;

    for scalar in list_elements {
        let Some(value) = scalar
            .as_primitive_opt()
            .and_then(|prim| prim.typed_value::<T>())
        else {
            if scalar.is_null() {
                has_null_element = true;
                continue;
            }
            return Ok(None);
        };

        let in_packed_domain = match range {
            Some((lo, hi)) => !value.is_lt(lo) && !value.is_gt(hi),
            None => true,
        };
        if in_packed_domain {
            if !in_domain[..in_domain_len].contains(&value) {
                if in_domain_len == MAX_FUSED_MEMBERS {
                    return Ok(None);
                }
                in_domain[in_domain_len] = value;
                in_domain_len += 1;
            }
        } else if has_patches {
            patch_only.push(value);
        }
    }

    let in_domain = &in_domain[..in_domain_len];
    let is_empty = list_elements.is_empty();

    // Fast path: zero in-domain members and no patches that could match.
    if in_domain.is_empty() && patch_only.is_empty() {
        return options
            .all_false_result(
                element.len(),
                element.validity()?,
                list_dtype,
                element.dtype(),
                is_empty,
                has_null_element,
            )
            .map(Some);
    }

    let use_binary_search = patch_only.len() > PATCH_BINARY_SEARCH_THRESHOLD;
    if use_binary_search {
        patch_only.sort_unstable();
        patch_only.dedup();
    }

    let exact = |value: T| {
        in_domain.contains(&value)
            || if use_binary_search {
                patch_only.binary_search(&value).is_ok()
            } else {
                patch_only.contains(&value)
            }
    };

    type V<T> = <T as PhysicalPType>::Physical;
    let bits = match in_domain {
        [] => constant_bits(element, false, &exact, ctx)?,
        [m0] => {
            let m0: V<T> = (*m0).as_();
            stream_compare_fused_bits::<T, V<T>, _>(
                element,
                bit_width,
                m0,
                V::<T>::is_eq,
                false,
                &exact,
                ctx,
            )?
        }
        [m0, m1] => {
            let (m0, m1): (V<T>, V<T>) = ((*m0).as_(), (*m1).as_());
            stream_compare_fused_bits::<T, V<T>, _>(
                element,
                bit_width,
                m0,
                move |v, r| v.is_eq(r) | v.is_eq(m1),
                false,
                &exact,
                ctx,
            )?
        }
        [m0, m1, m2] => {
            let (m0, m1, m2): (V<T>, V<T>, V<T>) = ((*m0).as_(), (*m1).as_(), (*m2).as_());
            stream_compare_fused_bits::<T, V<T>, _>(
                element,
                bit_width,
                m0,
                move |v, r| v.is_eq(r) | v.is_eq(m1) | v.is_eq(m2),
                false,
                &exact,
                ctx,
            )?
        }
        [m0, m1, m2, m3] => {
            let (m0, m1, m2, m3): (V<T>, V<T>, V<T>, V<T>) =
                ((*m0).as_(), (*m1).as_(), (*m2).as_(), (*m3).as_());
            stream_compare_fused_bits::<T, V<T>, _>(
                element,
                bit_width,
                m0,
                move |v, r| v.is_eq(r) | v.is_eq(m1) | v.is_eq(m2) | v.is_eq(m3),
                false,
                &exact,
                ctx,
            )?
        }
        _ => unreachable!("in_domain capped at MAX_FUSED_MEMBERS"),
    };

    let needle_validity = element.validity()?;
    if let Some(set) = prepared {
        return set
            .result_from_bits(bits, needle_validity, element.dtype(), options)
            .map(Some);
    }

    options
        .result_from_bits(
            bits,
            needle_validity,
            list_dtype,
            element.dtype(),
            is_empty,
            has_null_element,
        )
        .map(Some)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::lit;
    use vortex_expr::root;
    use vortex_functions::expr::list_contains_opts;
    use vortex_functions::list_contains::ListContains;
    use vortex_functions::list_contains::ListContainsOptions;
    use vortex_session::VortexSession;

    use crate::BitPackedData;
    use crate::FoRArraySlotsExt;
    use crate::bitpack_compress::bitpack_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        vortex_functions::initialize(&session);
        crate::initialize(&session);
        session
    });

    fn make_list_scalar(ptype: PType, values: &[Option<i64>], elem_null: Nullability) -> Scalar {
        let elem_dtype = Arc::new(DType::Primitive(ptype, elem_null));
        let scalars = values
            .iter()
            .map(|opt| match opt {
                Some(v) => match ptype {
                    PType::U8 => Scalar::primitive(u8::try_from(*v).unwrap(), elem_null),
                    PType::U16 => Scalar::primitive(u16::try_from(*v).unwrap(), elem_null),
                    PType::U32 => Scalar::primitive(u32::try_from(*v).unwrap(), elem_null),
                    PType::U64 => Scalar::primitive(u64::try_from(*v).unwrap(), elem_null),
                    PType::I8 => Scalar::primitive(i8::try_from(*v).unwrap(), elem_null),
                    PType::I16 => Scalar::primitive(i16::try_from(*v).unwrap(), elem_null),
                    PType::I32 => Scalar::primitive(i32::try_from(*v).unwrap(), elem_null),
                    PType::I64 => Scalar::primitive(*v, elem_null),
                    _ => unreachable!(),
                },
                None => Scalar::null(elem_dtype.as_ref().clone()),
            })
            .collect();
        Scalar::list(elem_dtype, scalars, Nullability::NonNullable)
    }

    fn eval_list_contains(
        array: &ArrayRef,
        list: Scalar,
        options: ListContainsOptions,
    ) -> VortexResult<ArrayRef> {
        let expr = list_contains_opts(lit(list), root(), options)
            .bind(array.dtype())?
            .optimize_recursive()?;
        array.clone().apply_bound(&expr)?.optimize_ctx(&SESSION)
    }

    #[rstest]
    #[case::m0(&[])]
    #[case::m1(&[Some(3)])]
    #[case::m2(&[Some(3), Some(11)])]
    #[case::m3(&[Some(3), Some(11), Some(27)])]
    #[case::m4(&[Some(3), Some(11), Some(27), Some(42)])]
    #[case::m4_with_dups(&[Some(3), Some(3), Some(11), Some(27), Some(42), Some(11)])]
    #[case::m5_fallback(&[Some(3), Some(11), Some(27), Some(42), Some(55)])]
    #[case::out_of_domain(&[Some(200), Some(300), Some(400)])]
    fn test_bitpacked_u32_members(#[case] members: &[Option<i64>]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter((0..2500u32).map(|i| i % 64));
        let packed = bitpack_encode(&prim, 6, None, &mut ctx)?.into_array();
        let expected_arr = prim.into_array();

        for sql_null_semantics in [false, true] {
            let opts = ListContainsOptions { sql_null_semantics };
            let list = make_list_scalar(PType::U32, members, Nullability::NonNullable);
            let actual = eval_list_contains(&packed, list.clone(), opts)?;
            let expected = eval_list_contains(&expected_arr, list, opts)?;
            let mut ctx = SESSION.create_execution_ctx();
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[rstest]
    #[case::u8(PType::U8, 5)]
    #[case::u16(PType::U16, 7)]
    #[case::u32(PType::U32, 6)]
    #[case::u64(PType::U64, 6)]
    #[case::i8(PType::I8, 5)]
    #[case::i16(PType::I16, 6)]
    #[case::i32(PType::I32, 6)]
    #[case::i64(PType::I64, 6)]
    fn test_bitpacked_all_integer_types(
        #[case] ptype: PType,
        #[case] bit_width: u8,
    ) -> VortexResult<()> {
        let len = 1200usize;
        let prim = match ptype {
            PType::U8 => PrimitiveArray::from_iter((0..len).map(|i| (i % 30) as u8)),
            PType::U16 => PrimitiveArray::from_iter((0..len).map(|i| (i % 100) as u16)),
            PType::U32 => PrimitiveArray::from_iter((0..len).map(|i| (i % 60) as u32)),
            PType::U64 => PrimitiveArray::from_iter((0..len).map(|i| (i % 60) as u64)),
            PType::I8 => PrimitiveArray::from_iter((0..len).map(|i| (i % 30) as i8)),
            PType::I16 => PrimitiveArray::from_iter((0..len).map(|i| (i % 60) as i16)),
            PType::I32 => PrimitiveArray::from_iter((0..len).map(|i| (i % 60) as i32)),
            PType::I64 => PrimitiveArray::from_iter((0..len).map(|i| (i % 60) as i64)),
            _ => unreachable!(),
        };
        let mut ctx = SESSION.create_execution_ctx();
        let packed = bitpack_encode(&prim, bit_width, None, &mut ctx)?.into_array();
        let expected_arr = prim.into_array();

        let members = if ptype.is_signed_int() {
            vec![Some(-5), Some(0), Some(7), Some(19), Some(120)]
        } else {
            vec![Some(0), Some(7), Some(19), Some(120)]
        };
        let list = make_list_scalar(ptype, &members, Nullability::NonNullable);
        let opts = ListContainsOptions {
            sql_null_semantics: true,
        };
        let actual = eval_list_contains(&packed, list.clone(), opts)?;
        let expected = eval_list_contains(&expected_arr, list, opts)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_bitpacked_sliced_and_patched() -> VortexResult<()> {
        let values: Vec<i32> = (0..2600i32)
            .map(|i| {
                if i % 101 == 0 {
                    200
                } else if i % 137 == 0 {
                    5000
                } else if i % 173 == 0 {
                    9999
                } else {
                    i % 32
                }
            })
            .collect();
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter(values);
        let packed = bitpack_encode(&prim, 5, None, &mut ctx)?.into_array();
        let expected_arr = prim.into_array();

        // Slice across chunk boundaries (offset > 0, len > 1024, non-multiple of 1024).
        let sliced_packed = packed.slice(37..2100)?;
        let sliced_expected = expected_arr.slice(37..2100)?;

        // Mixed in-domain + patch-only, patch-only (>8 to hit binary search), and all-miss.
        let test_sets: Vec<Vec<Option<i64>>> = vec![
            vec![Some(-10), Some(3), Some(15), Some(200), Some(5000)],
            vec![
                Some(-100),
                Some(-50),
                Some(-10),
                Some(200),
                Some(300),
                Some(400),
                Some(500),
                Some(600),
                Some(5000),
                Some(7000),
            ],
            vec![Some(-999), Some(8888)],
        ];

        for members in test_sets {
            let list = make_list_scalar(PType::I32, &members, Nullability::NonNullable);
            let opts = ListContainsOptions {
                sql_null_semantics: true,
            };
            let actual = eval_list_contains(&sliced_packed, list.clone(), opts)?;
            let expected = eval_list_contains(&sliced_expected, list, opts)?;
            let mut ctx = SESSION.create_execution_ctx();
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn test_bitpacked_bit_width_zero_and_empty() -> VortexResult<()> {
        let mut vals = vec![0u32; 500];
        vals[10] = 42;
        vals[250] = 99;
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter(vals);
        let packed = bitpack_encode(&prim, 0, None, &mut ctx)?.into_array();
        let expected_arr = prim.into_array();

        for members in [
            vec![Some(0)],
            vec![Some(42)],
            vec![Some(0), Some(99)],
            vec![Some(777)],
        ] {
            let list = make_list_scalar(PType::U32, &members, Nullability::NonNullable);
            let opts = ListContainsOptions {
                sql_null_semantics: false,
            };
            let actual = eval_list_contains(&packed, list.clone(), opts)?;
            let expected = eval_list_contains(&expected_arr, list, opts)?;
            let mut ctx = SESSION.create_execution_ctx();
            assert_arrays_eq!(actual, expected, &mut ctx);
        }

        // Empty slice (directly on BitPacked with len=0 via non-patched array, and sliced patched).
        let empty_packed = packed.slice(0..0)?;
        let empty_expected = expected_arr.slice(0..0)?;
        let list = make_list_scalar(PType::U32, &[Some(0), Some(42)], Nullability::NonNullable);
        let opts = ListContainsOptions {
            sql_null_semantics: true,
        };
        let actual = eval_list_contains(&empty_packed, list.clone(), opts)?;
        let expected = eval_list_contains(&empty_expected, list, opts)?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::sql_false(false)]
    #[case::sql_true(true)]
    fn test_bitpacked_null_semantics(#[case] sql_null_semantics: bool) -> VortexResult<()> {
        let opts = ListContainsOptions { sql_null_semantics };
        let values: Vec<Option<u32>> = (0..1100u32)
            .map(|i| if i % 7 == 0 { None } else { Some(i % 16) })
            .collect();
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_option_iter(values);
        let packed = bitpack_encode(&prim, 4, None, &mut ctx)?.into_array();
        let expected_arr = prim.into_array();

        let lists = vec![
            make_list_scalar(PType::U32, &[], Nullability::Nullable),
            make_list_scalar(PType::U32, &[None], Nullability::Nullable),
            make_list_scalar(
                PType::U32,
                &[Some(3), None, Some(11)],
                Nullability::Nullable,
            ),
            make_list_scalar(PType::U32, &[Some(100), None], Nullability::Nullable),
            make_list_scalar(
                PType::U32,
                &[Some(100), Some(200)],
                Nullability::NonNullable,
            ),
            Scalar::null(DType::List(
                Arc::new(DType::Primitive(PType::U32, Nullability::Nullable)),
                Nullability::Nullable,
            )),
        ];

        for list in lists {
            let actual = eval_list_contains(&packed, list.clone(), opts)?;
            let expected = eval_list_contains(&expected_arr, list, opts)?;
            let mut ctx = SESSION.create_execution_ctx();
            assert_arrays_eq!(actual, expected, &mut ctx);
        }

        // Unprepared Constant list directly via ListContains (bypassing PreparedSet).
        let raw_list = make_list_scalar(
            PType::U32,
            &[Some(3), None, Some(11)],
            Nullability::Nullable,
        );
        let const_list = ConstantArray::new(raw_list, packed.len()).into_array();
        let actual = ListContains::try_new_opts(const_list.clone(), packed, opts)?.into_array();
        let expected = ListContains::try_new_opts(const_list, expected_arr, opts)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_bitpacked_kernel_dispatch_trace() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter((0..1024u32).map(|i| i % 32));
        let packed = bitpack_encode(&prim, 5, None, &mut ctx)?.into_array();
        let opts = ListContainsOptions {
            sql_null_semantics: false,
        };

        // 0 in-domain (out-of-domain, no patches) -> kernel returns ConstantArray(false).
        let ood = eval_list_contains(
            &packed,
            make_list_scalar(
                PType::U32,
                &[Some(100), Some(200)],
                Nullability::NonNullable,
            ),
            opts,
        )?;
        let executed = ctx
            .try_execute_parent(&ood)?
            .expect("kernel handles 0 in-domain members");
        assert!(executed.is::<vortex_array::arrays::Constant>());

        // 4 distinct in-domain (with duplicates) -> kernel handles it.
        let m4 = eval_list_contains(
            &packed,
            make_list_scalar(
                PType::U32,
                &[Some(1), Some(5), Some(12), Some(20), Some(5)],
                Nullability::NonNullable,
            ),
            opts,
        )?;
        assert!(
            ctx.try_execute_parent(&m4)?.is_some(),
            "kernel handles <= 4 distinct in-domain members"
        );

        // 5 distinct in-domain -> kernel declines (returns None) to fall back to PreparedSet.
        let m5 = eval_list_contains(
            &packed,
            make_list_scalar(
                PType::U32,
                &[Some(1), Some(5), Some(12), Some(20), Some(25)],
                Nullability::NonNullable,
            ),
            opts,
        )?;
        assert!(
            ctx.try_execute_parent(&m5)?.is_none(),
            "kernel declines >= 5 distinct in-domain members"
        );

        // Blocked bit widths -> kernel declines.
        let prim_arr = prim.into_array();
        let blocked = BitPackedData::encode_blocked(&prim_arr, &[5u8], &mut ctx)?.into_array();
        let blocked_expr = eval_list_contains(
            &blocked,
            make_list_scalar(PType::U32, &[Some(1), Some(5)], Nullability::NonNullable),
            opts,
        )?;
        assert!(
            ctx.try_execute_parent(&blocked_expr)?.is_none(),
            "kernel declines blocked bit widths"
        );

        // Make sure blocked still executes correctly via fallback.
        let expected = eval_list_contains(
            &prim_arr,
            make_list_scalar(PType::U32, &[Some(1), Some(5)], Nullability::NonNullable),
            opts,
        )?;
        assert_arrays_eq!(blocked_expr, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::non_sql(false)]
    #[case::sql(true)]
    fn test_bitpacked_boundary_and_combined_edge_cases(
        #[case] sql_null_semantics: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let opts = ListContainsOptions { sql_null_semantics };

        // Boundary members: 2^W - 1 (= 15) and 2^W (= 16), plus T::MIN / T::MAX.
        let values: Vec<Option<i32>> = (0..2200)
            .map(|i| match i % 11 {
                0 => None,
                1 => Some(15),       // 2^4 - 1 (largest packed value)
                2 => Some(16),       // 2^4 (first patch-only value)
                3 => Some(i32::MAX), // T::MAX patch
                _ => Some(i % 15),
            })
            .collect();
        let prim = PrimitiveArray::from_option_iter(values);
        let bp = bitpack_encode(&prim, 4, None, &mut ctx)?.into_array();
        // Nullable + patched + sliced across chunk boundaries.
        let sliced_bp = bp.slice(13..2077)?;
        let sliced_prim = prim.into_array().slice(13..2077)?;

        for members in [
            vec![Some(15i32), Some(16), Some(i32::MIN), Some(i32::MAX)],
            vec![Some(15i32), Some(16), None],
            vec![Some(16i32), Some(i32::MIN), Some(i32::MAX)],
            vec![Some(i32::MIN), Some(i32::MAX), None],
        ] {
            let list = Scalar::list(
                Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                members
                    .into_iter()
                    .map(|m| match m {
                        Some(v) => Scalar::primitive(v, Nullability::Nullable),
                        None => Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                    })
                    .collect(),
                Nullability::NonNullable,
            );
            let actual = eval_list_contains(&sliced_bp, list.clone(), opts)?;
            let expected = eval_list_contains(&sliced_prim, list, opts)?;
            assert_eq!(actual.dtype(), expected.dtype());
            assert_arrays_eq!(actual, expected, &mut ctx);
        }

        // FoR over patched BitPacked + FoR shift overflow on query members (reference = i32::MIN).
        let for_bp_values: Vec<Option<i32>> = (0..1500)
            .map(|i| match i % 13 {
                0 => None,
                1 => Some(i32::MIN),
                2 => Some(i32::MIN + 15),
                3 => Some(i32::MIN + 16), // forces patch in bit_width=4 child
                _ => Some(i32::MIN + (i % 14)),
            })
            .collect();
        let for_bp_prim = PrimitiveArray::from_option_iter(for_bp_values);
        let for_bp_arr = crate::FoRData::encode(for_bp_prim.clone(), &mut ctx)?;
        let encoded_child = for_bp_arr
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let patched_bp = bitpack_encode(&encoded_child, 4, None, &mut ctx)?.into_array();
        let for_over_bp = crate::FoR::try_new(
            patched_bp,
            Scalar::primitive(i32::MIN, Nullability::Nullable),
        )?
        .into_array()
        .slice(7..1400)?;
        let for_bp_prim_sliced = for_bp_prim.into_array().slice(7..1400)?;

        // FoR with extreme spread (i32::MIN .. i32::MAX) where wrapping_sub overflows signed i32.
        let for_overflow_prim = PrimitiveArray::from_option_iter([
            Some(i32::MIN),
            Some(i32::MIN + 1),
            Some(0),
            Some(i32::MAX),
            None,
        ]);
        let for_overflow =
            crate::FoRData::encode(for_overflow_prim.clone(), &mut ctx)?.into_array();
        let for_overflow_expected = for_overflow_prim.into_array();

        for members in [
            vec![
                Some(i32::MIN),
                Some(i32::MIN + 15),
                Some(i32::MIN + 16),
                Some(i32::MAX),
            ],
            vec![Some(i32::MIN + 16), Some(i32::MAX), None],
            vec![Some(i32::MAX)],
        ] {
            let list = Scalar::list(
                Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                members
                    .into_iter()
                    .map(|m| match m {
                        Some(v) => Scalar::primitive(v, Nullability::Nullable),
                        None => Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                    })
                    .collect(),
                Nullability::NonNullable,
            );
            let actual = eval_list_contains(&for_over_bp, list.clone(), opts)?;
            let expected = eval_list_contains(&for_bp_prim_sliced, list.clone(), opts)?;
            assert_eq!(actual.dtype(), expected.dtype());
            assert_arrays_eq!(actual, expected, &mut ctx);

            let actual_ovf = eval_list_contains(&for_overflow, list.clone(), opts)?;
            let expected_ovf = eval_list_contains(&for_overflow_expected, list, opts)?;
            assert_eq!(actual_ovf.dtype(), expected_ovf.dtype());
            assert_arrays_eq!(actual_ovf, expected_ovf, &mut ctx);
        }
        Ok(())
    }

    fn next_rand(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn test_bitpacked_and_for_seeded_differential() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = 0x5EED_C0DE_1234_5678u64;

        let ptypes = [
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
        ];

        for ptype in ptypes {
            let max_bw = (ptype.byte_width() * 8) as u8;
            let bit_widths = [0u8, 1, 3, 4, 7.min(max_bw - 1), max_bw - 1];

            for &bw in &bit_widths {
                let packed_max: u64 = if bw == 0 { 0 } else { (1u64 << bw) - 1 };
                let len = (next_rand(&mut rng) % 1300) as usize + 1;
                let nullable = (next_rand(&mut rng) & 1) == 1;
                let is_signed = ptype.is_signed_int();
                let max_pos: u64 = if is_signed {
                    (1u64 << (max_bw - 1)) - 1
                } else if max_bw == 64 {
                    u64::MAX
                } else {
                    (1u64 << max_bw) - 1
                };

                let raw: Vec<Option<u64>> = (0..len)
                    .map(|_| {
                        if nullable && next_rand(&mut rng).is_multiple_of(7) {
                            None
                        } else if next_rand(&mut rng).is_multiple_of(20) {
                            Some(next_rand(&mut rng) % max_pos.max(1))
                        } else {
                            Some(if packed_max == 0 {
                                0
                            } else {
                                next_rand(&mut rng) % (packed_max + 1)
                            })
                        }
                    })
                    .collect();

                let prim = match ptype {
                    PType::U8 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as u8)))
                    }
                    PType::U16 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as u16)))
                    }
                    PType::U32 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as u32)))
                    }
                    PType::U64 => PrimitiveArray::from_option_iter(raw.iter().copied()),
                    PType::I8 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as i8)))
                    }
                    PType::I16 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as i16)))
                    }
                    PType::I32 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as i32)))
                    }
                    PType::I64 => {
                        PrimitiveArray::from_option_iter(raw.iter().map(|v| v.map(|x| x as i64)))
                    }
                    _ => unreachable!(),
                };
                let bp = bitpack_encode(&prim, bw, None, &mut ctx)?.into_array();

                let slice_start = (next_rand(&mut rng) as usize) % len;
                let slice_end =
                    slice_start + ((next_rand(&mut rng) as usize) % (len - slice_start + 1));
                let bp_sliced = bp.slice(slice_start..slice_end)?;
                let prim_sliced = prim.into_array().slice(slice_start..slice_end)?;

                let set_len = (next_rand(&mut rng) % 7) as usize;
                let list_has_null = (next_rand(&mut rng) & 1) == 1;
                let elem_nullability = if list_has_null {
                    Nullability::Nullable
                } else {
                    Nullability::NonNullable
                };
                let mut members: Vec<Scalar> = (0..set_len)
                    .map(|i| {
                        let v = match i {
                            0 => packed_max.min(max_pos),
                            1 => packed_max.saturating_add(1).min(max_pos),
                            _ => {
                                next_rand(&mut rng)
                                    % (packed_max.saturating_add(4)).min(max_pos).max(1)
                            }
                        };
                        match ptype {
                            PType::U8 => Scalar::primitive(v as u8, elem_nullability),
                            PType::U16 => Scalar::primitive(v as u16, elem_nullability),
                            PType::U32 => Scalar::primitive(v as u32, elem_nullability),
                            PType::U64 => Scalar::primitive(v, elem_nullability),
                            PType::I8 => Scalar::primitive(v as i8, elem_nullability),
                            PType::I16 => Scalar::primitive(v as i16, elem_nullability),
                            PType::I32 => Scalar::primitive(v as i32, elem_nullability),
                            PType::I64 => Scalar::primitive(v as i64, elem_nullability),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                if list_has_null {
                    members.push(Scalar::null(DType::Primitive(ptype, Nullability::Nullable)));
                }
                let list_scalar = Scalar::list(
                    Arc::new(DType::Primitive(ptype, elem_nullability)),
                    members,
                    Nullability::NonNullable,
                );

                for sql_null_semantics in [false, true] {
                    let opts = ListContainsOptions { sql_null_semantics };
                    let actual = eval_list_contains(&bp_sliced, list_scalar.clone(), opts)?;
                    let expected = eval_list_contains(&prim_sliced, list_scalar.clone(), opts)?;
                    assert_eq!(actual.dtype(), expected.dtype());
                    assert_arrays_eq!(actual, expected, &mut ctx);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_large_set_falls_back_before_extracting_elements() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter(0u32..128);
        let bp = BitPackedData::encode(&prim.into_array(), 7, &mut ctx)?;
        // Out-of-domain members so only MAX_SET_LEN triggers fallback.
        let list = Scalar::list(
            Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable)),
            (200u32..200 + (super::MAX_SET_LEN as u32) + 1)
                .map(Scalar::from)
                .collect(),
            Nullability::NonNullable,
        );
        let prepared = super::PreparedSet::try_new_array(list, bp.len())?;
        let res = <super::BitPacked as super::ListContainsElementKernel>::list_contains(
            &prepared,
            bp.as_view(),
            &ListContainsOptions::default(),
            &mut ctx,
        )?;
        assert!(res.is_none());
        Ok(())
    }
}
