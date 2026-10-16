// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use smallvec::smallvec;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use super::ArrayHaystack;
use super::HaystackVTable;
use super::HaystackValidity;
use super::bytes::BytesHaystack;
use super::haystack_vtable_common;
use super::integer::BitmapHaystack;
use super::integer::SortedHaystack;
use super::rows::RowsHaystack;
use super::slot;
use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::CanonicalView;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::PreparedSet;
use crate::scalar_fn::fns::list_contains::PreparedSetData;
use crate::validity::Validity;

/// A struct array with one `i32` field of `values`.
fn struct_of(values: &[i32]) -> ArrayRef {
    StructArray::try_new(
        FieldNames::from(["a"]),
        vec![PrimitiveArray::from_iter(values.iter().copied()).into_array()],
        values.len(),
        Validity::NonNullable,
    )
    .unwrap()
    .into_array()
}

fn build(elements: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayHaystack> {
    ArrayHaystack::try_new(&elements, ctx)
}

/// Each kind of element goes into its lookup array, which canonicalizes to the distinct non-null
/// elements in the order of the lookup.
#[rstest]
#[case::dense_integers(
    PrimitiveArray::from_option_iter([Some(5i64), Some(-3), None, Some(5)]).into_array(),
    BitmapHaystack.id(),
    PrimitiveArray::from_iter([-3i64, 5]).into_array(),
)]
#[case::sparse_integers(
    PrimitiveArray::from_iter([1i64 << 40, 7, -(1i64 << 40), 7]).into_array(),
    SortedHaystack.id(),
    PrimitiveArray::from_iter([-(1i64 << 40), 7, 1i64 << 40]).into_array(),
)]
#[case::floats_by_bits(
    PrimitiveArray::from_iter([1.5f64, -0.0, 0.0]).into_array(),
    SortedHaystack.id(),
    PrimitiveArray::from_iter([0.0f64, 1.5, -0.0]).into_array(),
)]
#[case::dense_decimals(
    DecimalArray::from_option_iter::<i32, _>([Some(12), None, Some(10)], DecimalDType::new(5, 1))
        .into_array(),
    BitmapHaystack.id(),
    DecimalArray::from_iter::<i32, _>([10, 12], DecimalDType::new(5, 1)).into_array(),
)]
#[case::rows(
    struct_of(&[3, 1, 3]),
    RowsHaystack.id(),
    struct_of(&[1, 3]),
)]
fn test_lookup_array_of_each_kind(
    #[case] elements: ArrayRef,
    #[case] encoding: ArrayId,
    #[case] canonical: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let haystack = build(elements, &mut ctx)?;

    assert_eq!(haystack.array().encoding_id(), encoding);
    assert_arrays_eq!(haystack.array().clone(), canonical, &mut ctx);
    Ok(())
}

#[test]
fn test_bytes_lookup_array_keeps_the_non_null_elements() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let elements =
        VarBinViewArray::from_iter_nullable_str([Some("a"), None, Some("a long string value")])
            .into_array();
    let haystack = build(elements, &mut ctx)?;

    assert_eq!(haystack.array().encoding_id(), BytesHaystack.id());
    assert_arrays_eq!(
        haystack.array().clone(),
        VarBinViewArray::from_iter_nullable_str([Some("a"), Some("a long string value")]),
        &mut ctx
    );

    let needles = VarBinViewArray::from_iter_nullable_str([
        Some("a long string value"),
        Some("a long string valuf"),
        Some("a"),
        None,
    ])
    .into_array();
    let (bits, validity) = haystack.contains(needles.as_::<AnyCanonical>(), &mut ctx)?;
    assert_eq!(bits, BitBuffer::from_iter([true, false, true, false]));
    assert_eq!(validity.execute_mask(4, &mut ctx)?.true_count(), 3);
    Ok(())
}

/// Each kind of lookup array reads a row in the order of its lookup.
#[rstest]
#[case::bitmap(buffer![9u8, 2, 4].into_array(), Scalar::from(4u8))]
#[case::sorted(buffer![1u64 << 40, 7, 1 << 41].into_array(), Scalar::from(1u64 << 40))]
#[case::decimal_bitmap(
    DecimalArray::from_iter::<i32, _>([12, 10, 11], DecimalDType::new(5, 1)).into_array(),
    Scalar::decimal(11i32.into(), DecimalDType::new(5, 1), Nullability::NonNullable),
)]
#[case::bytes(
    VarBinViewArray::from_iter_str(["b", "a long string value"]).into_array(),
    Scalar::from("a long string value"),
)]
fn test_lookup_array_reads_a_row(
    #[case] elements: ArrayRef,
    #[case] expected: Scalar,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let haystack = build(elements, &mut ctx)?;

    assert_eq!(haystack.array().execute_scalar(1, &mut ctx)?, expected);
    Ok(())
}

#[test]
fn test_rows_lookup_array_reads_a_row() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let haystack = build(struct_of(&[3, 1, 3]), &mut ctx)?;

    assert_eq!(
        haystack.array().execute_scalar(1, &mut ctx)?,
        struct_of(&[3]).execute_scalar(0, &mut ctx)?
    );
    Ok(())
}

/// A lookup array of your own encoding: `i32` values found by a linear scan.
#[derive(Clone, Debug)]
struct LinearHaystack;

impl LinearHaystack {
    fn new(values: ArrayRef) -> Array<Self> {
        Array::try_from_parts(ArrayParts::new(
            Self,
            values.dtype().clone(),
            values.len(),
            EmptyArrayData,
            smallvec![Some(values)],
        ))
        .unwrap()
    }
}

impl VTable for LinearHaystack {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = HaystackValidity;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("test.haystack.linear");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        _dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        Ok(())
    }

    haystack_vtable_common!("LinearHaystack");

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        "values".to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(slot(array.slots(), 0).clone()))
    }
}

impl OperationsVTable<LinearHaystack> for LinearHaystack {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        slot(array.slots(), 0).execute_scalar(index, ctx)
    }
}

impl HaystackVTable for LinearHaystack {
    fn contains(
        array: ArrayView<'_, Self>,
        needles: CanonicalView<'_>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let CanonicalView::Primitive(needles) = needles else {
            vortex_bail!("A linear haystack looks up primitive needles");
        };
        let values = slot(array.slots(), 0).as_::<Primitive>();
        let bits = needles
            .as_slice::<i32>()
            .iter()
            .map(|needle| values.as_slice::<i32>().contains(needle))
            .collect();
        Ok((bits, needles.validity()?))
    }
}

fn i32_list(values: &[i32]) -> Scalar {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    Scalar::list(
        std::sync::Arc::new(dtype),
        values.iter().map(|&value| Scalar::from(value)).collect(),
        Nullability::NonNullable,
    )
}

#[test]
fn test_prepared_set_looks_up_through_your_own_haystack() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // The lookup array holds 7 only, so the set finds 7 and not 3, although the list holds both.
    let haystack = ArrayHaystack::new(LinearHaystack::new(buffer![7i32].into_array()));
    let set = PreparedSetData::try_with_haystack(i32_list(&[3, 7]), haystack)?;

    assert_eq!(
        set.haystack(&mut ctx)?.array().encoding_id(),
        LinearHaystack.id()
    );
    assert_arrays_eq!(
        set.contains(
            &buffer![7i32, 3, 1].into_array(),
            &ListContainsOptions::default(),
            &mut ctx
        )?,
        BoolArray::from_iter([true, false, false]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_list_contains_executes_through_your_own_haystack() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let haystack = ArrayHaystack::new(LinearHaystack::new(buffer![7i32].into_array()));
    let set = PreparedSetData::try_with_haystack(i32_list(&[3, 7]), haystack)?;
    let result = ListContains::try_new(
        PreparedSet::new_array(set, 3),
        buffer![7i32, 3, 1].into_array(),
    )?
    .into_array();

    assert_arrays_eq!(result, BoolArray::from_iter([true, false, false]), &mut ctx);
    Ok(())
}

#[test]
fn test_your_own_haystack_must_hold_the_element_dtype() {
    let haystack = ArrayHaystack::new(LinearHaystack::new(buffer![7i64].into_array()));
    assert!(PreparedSetData::try_with_haystack(i32_list(&[7]), haystack).is_err());
}

#[test]
fn test_a_built_haystack_canonicalizes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let haystack = build(buffer![1u16, 1, 3].into_array(), &mut ctx)?;
    let canonical = haystack.array().clone().execute::<Canonical>(&mut ctx)?;
    assert_arrays_eq!(
        canonical.into_array(),
        buffer![1u16, 3].into_array(),
        &mut ctx
    );
    Ok(())
}
