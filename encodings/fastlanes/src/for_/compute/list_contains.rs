// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ListContainsElementReduce` rule for [`FoR`].
//!
//! Shifts constant set members down by the global FoR reference, drops members strictly below the
//! reference (no valid FoR value can match them), preserves a null member if present for SQL null
//! semantics, and rewrites `ListContains` onto the encoded child array.

use std::sync::Arc;

use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::list_contains::ListContains;
use vortex_functions::list_contains::ListContainsElementReduce;
use vortex_functions::list_contains::ListContainsOptions;
use vortex_functions::list_contains::PreparedSet;
use vortex_functions::list_contains::PreparedSetData;

use crate::FoR;
use crate::FoRReferences;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

/// Bail out on large `IN` sets so chunked scans reuse the shared `PreparedSet` haystack instead
/// of shifting and rebuilding a new `PreparedSet` per FoR chunk.
const MAX_SET_LEN: usize = 64;

impl ListContainsElementReduce for FoR {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let FoRReferences::Global(reference) = element.references() else {
            return Ok(None);
        };

        let Some(list_scalar) = PreparedSet::set_of(list)
            .map(PreparedSetData::list)
            .or_else(|| {
                list.as_opt::<Constant>()
                    .map(|constant| constant.data().scalar())
            })
        else {
            return Ok(None);
        };

        let DType::List(member_dtype, list_nullability) = list_scalar.dtype() else {
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
            reduce_for_list_contains::<T>(
                element,
                reference,
                member_dtype,
                *list_nullability,
                &list_elements,
                options,
            )
        })
    }
}

fn reduce_for_list_contains<T>(
    element: ArrayView<'_, FoR>,
    reference: &Scalar,
    member_dtype: &Arc<DType>,
    list_nullability: Nullability,
    list_elements: &[Scalar],
    options: &ListContainsOptions,
) -> VortexResult<Option<ArrayRef>>
where
    T: IntegerPType + WrappingSub + TryFrom<PValue, Error = VortexError>,
    PValue: From<T>,
{
    let reference_val = reference
        .as_primitive()
        .typed_value::<T>()
        .vortex_expect("FoR reference is non-null and matches the array ptype");

    let member_nullability = member_dtype.nullability();
    let mut shifted = Vec::with_capacity(list_elements.len());
    let mut has_null = false;

    for scalar in list_elements {
        let Some(value) = scalar
            .as_primitive_opt()
            .and_then(|prim| prim.typed_value::<T>())
        else {
            if scalar.is_null() {
                has_null = true;
                continue;
            }
            return Ok(None);
        };

        // No valid FoR value is below the reference.
        if value < reference_val {
            continue;
        }
        shifted.push(Scalar::primitive(
            value.wrapping_sub(&reference_val),
            member_nullability,
        ));
    }

    // No valid row can match any element.
    if shifted.is_empty() {
        let list_dtype = DType::List(Arc::clone(member_dtype), list_nullability);
        return options
            .all_false_result(
                element.len(),
                element.validity()?,
                &list_dtype,
                element.dtype(),
                list_elements.is_empty(),
                has_null,
            )
            .map(Some);
    }

    if has_null {
        shifted.push(Scalar::null(member_dtype.as_ref().clone()));
    }

    let shifted_list = Scalar::list(Arc::clone(member_dtype), shifted, list_nullability);
    let prepared_list = PreparedSet::try_new_array(shifted_list, element.len())?;
    Ok(Some(
        ListContains::try_new_opts(prepared_list, element.encoded().clone(), *options)?
            .into_array(),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use hegel::TestCase;
    use hegel::generators as gs;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFn;
    use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::hegel::draw_validity_for;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::lit;
    use vortex_expr::root;
    use vortex_functions::expr::list_contains_opts;
    use vortex_functions::list_contains::ListContainsOptions;
    use vortex_session::VortexSession;

    use crate::FoR;
    use crate::FoRData;
    use crate::bitpack_compress::bitpack_encode;
    use crate::for_::array::FoRArraySlotsExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        vortex_functions::initialize(&session);
        crate::initialize(&session);
        session
    });

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
    #[case::mixed(vec![Some(900), Some(1000), Some(1005), Some(1012)])]
    #[case::all_below_ref(vec![Some(100), Some(500), Some(999)])]
    #[case::with_null_and_match(vec![Some(900), None, Some(1005)])]
    #[case::all_below_ref_with_null(vec![Some(100), None, Some(999)])]
    #[case::empty(vec![])]
    fn test_for_bitpacked_i32(#[case] members: Vec<Option<i32>>) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter((0..1200i32).map(|i| 1000 + (i % 32)));
        let for_arr = FoRData::encode(prim.clone(), &mut ctx)?;
        let packed_child = bitpack_encode(
            &for_arr
                .encoded()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?,
            5,
            None,
            &mut ctx,
        )?;
        let for_bp = FoR::try_new(
            packed_child.into_array(),
            Scalar::primitive(1000i32, Nullability::NonNullable),
        )?
        .into_array();
        let expected_arr = prim.into_array();

        let elem_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::Nullable));
        let scalars = members
            .into_iter()
            .map(|opt| match opt {
                Some(v) => Scalar::primitive(v, Nullability::Nullable),
                None => Scalar::null(elem_dtype.as_ref().clone()),
            })
            .collect();
        let list = Scalar::list(elem_dtype, scalars, Nullability::NonNullable);

        for sql_null_semantics in [false, true] {
            let opts = ListContainsOptions {
                sql_null_semantics,
                ..Default::default()
            };
            let actual = eval_list_contains(&for_bp, list.clone(), opts)?;
            let expected = eval_list_contains(&expected_arr, list.clone(), opts)?;
            // FoR reduce rule should have eliminated the FoR wrapper.
            assert!(
                !actual.is::<FoR>()
                    && !actual
                        .as_opt::<ScalarFn>()
                        .is_some_and(|sfn| sfn.children().iter().any(|c| c.is::<FoR>())),
                "FoR should be reduced away"
            );
            let mut ctx = SESSION.create_execution_ctx();
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn test_for_all_below_ref_reduces_to_constant() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter((0..500u32).map(|i| 10_000 + (i % 16)));
        let for_arr = FoRData::encode(prim, &mut ctx)?.into_array();
        let elem_dtype = Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable));
        let list = Scalar::list(
            elem_dtype,
            vec![
                Scalar::primitive(1u32, Nullability::NonNullable),
                Scalar::primitive(9_999u32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );
        let reduced = eval_list_contains(
            &for_arr,
            list,
            ListContainsOptions {
                sql_null_semantics: true,
                ..Default::default()
            },
        )?;
        assert!(reduced.is::<Constant>());
        assert_eq!(
            reduced.as_constant(),
            Some(&Scalar::bool(false, Nullability::NonNullable))
        );
        Ok(())
    }

    #[test]
    fn test_for_nullable_and_signed_wrapping() -> VortexResult<()> {
        // Signed wrapping distance: reference = -100i8, value = 100i8 -> distance = 200u8 as i8 = -56i8.
        let values: Vec<Option<i8>> = (0..200)
            .map(|i| match i % 5 {
                0 => None,
                1 => Some(-100),
                2 => Some(0),
                3 => Some(100),
                _ => Some(-50),
            })
            .collect();
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_option_iter(values);
        let for_arr = FoRData::encode(prim.clone(), &mut ctx)?.into_array();
        let expected_arr = prim.into_array();

        let elem_dtype = Arc::new(DType::Primitive(PType::I8, Nullability::Nullable));
        for members in [
            vec![Some(-120i8), Some(-100), Some(100), None],
            vec![Some(-120i8), Some(-110)],
            vec![Some(-120i8), None],
        ] {
            let scalars = members
                .into_iter()
                .map(|opt| match opt {
                    Some(v) => Scalar::primitive(v, Nullability::Nullable),
                    None => Scalar::null(elem_dtype.as_ref().clone()),
                })
                .collect();
            let list = Scalar::list(Arc::clone(&elem_dtype), scalars, Nullability::NonNullable);
            for sql_null_semantics in [false, true] {
                let opts = ListContainsOptions {
                    sql_null_semantics,
                    ..Default::default()
                };
                let actual = eval_list_contains(&for_arr, list.clone(), opts)?;
                let expected = eval_list_contains(&expected_arr, list.clone(), opts)?;
                let mut ctx = SESSION.create_execution_ctx();
                assert_arrays_eq!(actual, expected, &mut ctx);
            }
        }
        Ok(())
    }

    #[hegel::test(test_cases = 50, print_blob = true)]
    fn test_for_list_contains_differential(tc: TestCase) {
        const PTYPES: [PType; 8] = [
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
        ];
        let mut ctx = SESSION.create_execution_ctx();
        let ptype = tc.draw(gs::sampled_from(&PTYPES));
        let len = tc.draw(gs::integers::<usize>().max_value(64));
        let nullability = if tc.draw(gs::booleans()) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let validity = draw_validity_for(&tc, nullability, len);
        let list_has_null = tc.draw(gs::booleans());
        let elem_nullability = if list_has_null {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let set_len = tc.draw(gs::integers::<usize>().max_value(6));

        macro_rules! draw_prim_and_members {
            ($ty:ty) => {{
                let vals = tc.draw(gs::vecs(gs::integers::<$ty>()).min_size(len).max_size(len));
                let mut members: Vec<Scalar> = (0..set_len)
                    .map(|_| {
                        let v: $ty = if !vals.is_empty() && tc.draw(gs::booleans()) {
                            let base = vals[tc.draw(gs::integers::<usize>().max_value(len - 1))];
                            match tc.draw(gs::integers::<u8>().max_value(3)) {
                                0 => base,
                                1 => base.wrapping_sub(1),
                                2 => base.wrapping_add(1),
                                _ => tc.draw(gs::sampled_from(&[<$ty>::MIN, <$ty>::MAX])),
                            }
                        } else {
                            tc.draw(gs::integers::<$ty>())
                        };
                        Scalar::primitive(v, elem_nullability)
                    })
                    .collect();
                let prim = PrimitiveArray::new(Buffer::from(vals), validity);
                if list_has_null {
                    members.push(Scalar::null(DType::Primitive(ptype, Nullability::Nullable)));
                }
                (prim, members)
            }};
        }

        let (prim, members) = match ptype {
            PType::U8 => draw_prim_and_members!(u8),
            PType::U16 => draw_prim_and_members!(u16),
            PType::U32 => draw_prim_and_members!(u32),
            PType::U64 => draw_prim_and_members!(u64),
            PType::I8 => draw_prim_and_members!(i8),
            PType::I16 => draw_prim_and_members!(i16),
            PType::I32 => draw_prim_and_members!(i32),
            PType::I64 => draw_prim_and_members!(i64),
            _ => unreachable!(),
        };

        let has_valid = prim
            .validity()
            .vortex_expect("prim validity")
            .execute_mask(len, &mut ctx)
            .vortex_expect("validity mask")
            .true_count()
            > 0;
        let for_arr = if has_valid {
            FoRData::encode(prim.clone(), &mut ctx)
                .vortex_expect("FoR::encode")
                .into_array()
        } else {
            let zero = Scalar::default_value(&DType::Primitive(ptype, Nullability::NonNullable))
                .cast(prim.dtype())
                .vortex_expect("zero ref");
            FoR::try_new(prim.clone().into_array(), zero)
                .vortex_expect("FoR::try_new")
                .into_array()
        };
        let list = Scalar::list(
            Arc::new(DType::Primitive(ptype, elem_nullability)),
            members,
            Nullability::NonNullable,
        );
        let opts = ListContainsOptions {
            sql_null_semantics: tc.draw(gs::booleans()),
            ..Default::default()
        };

        let actual = eval_list_contains(&for_arr, list.clone(), opts)
            .vortex_expect("eval FoR list_contains");
        let expected = eval_list_contains(&prim.into_array(), list, opts)
            .vortex_expect("eval prim list_contains");
        assert_eq!(actual.dtype(), expected.dtype());
        assert_arrays_eq!(actual, expected, &mut ctx);
    }

    #[test]
    fn test_for_large_set_falls_back() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter(1000i32..1128);
        let for_arr = FoRData::encode(prim, &mut ctx)?;
        let list = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            (0..(super::MAX_SET_LEN as i32) + 1)
                .map(|v| Scalar::from(1000 + v))
                .collect(),
            Nullability::NonNullable,
        );
        let prepared = super::PreparedSet::try_new_array(list, for_arr.len())?;
        let res = <FoR as super::ListContainsElementReduce>::list_contains(
            &prepared,
            for_arr.as_view(),
            &ListContainsOptions::default(),
        )?;
        assert!(res.is_none());
        Ok(())
    }
}
