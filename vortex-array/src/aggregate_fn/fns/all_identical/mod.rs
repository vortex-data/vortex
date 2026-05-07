// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::arrays::DecimalArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::StructArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::list_from_list_view;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Check if two arrays are element-wise identical, treating null == null as true.
///
/// Returns `true` if and only if:
/// - Both arrays have the same dtype and length
/// - At every position, both are null or both are non-null with the same value
///
/// This is more efficient than element-wise `execute_scalar` comparison because it
/// operates on buffers directly and can short-circuit on the first mismatch.
pub fn all_identical(a: &ArrayRef, b: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    if a.dtype() != b.dtype() {
        vortex_bail!(
            "all_identical: dtype mismatch: {} vs {}",
            a.dtype(),
            b.dtype()
        );
    }

    if a.len() != b.len() {
        vortex_bail!("all_identical: length mismatch: {} vs {}", a.len(), b.len());
    }

    if a.is_empty() {
        return Ok(true);
    }

    let struct_dtype = make_all_identical_input_dtype(a.dtype());
    let struct_array = StructArray::try_new(
        NAMES.clone(),
        vec![a.clone(), b.clone()],
        a.len(),
        Validity::NonNullable,
    )?;

    let mut acc = Accumulator::try_new(AllIdentical, EmptyOptions, struct_dtype)?;
    acc.accumulate(&struct_array.into_array(), ctx)?;
    let result = acc.finish()?;

    Ok(result.as_bool().value().unwrap_or(false))
}

static NAMES: std::sync::LazyLock<FieldNames> =
    std::sync::LazyLock::new(|| FieldNames::from(["lhs", "rhs"]));

fn make_all_identical_input_dtype(element_dtype: &DType) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![element_dtype.clone(), element_dtype.clone()],
        ),
        Nullability::NonNullable,
    )
}

/// Aggregation function that checks if two arrays are element-wise identical.
///
/// The input is a `Struct{lhs: T, rhs: T}` and the result is `Bool(NonNullable)`.
#[derive(Clone, Debug)]
pub struct AllIdentical;

/// Partial accumulator state: just a bool tracking "all identical so far".
pub struct AllIdenticalPartial {
    all_identical: bool,
}

impl AggregateFnVTable for AllIdentical {
    type Options = EmptyOptions;
    type Partial = AllIdenticalPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.all_identical")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("AllIdentical is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // Input must be a struct with exactly two fields of the same type.
        match input_dtype {
            DType::Struct(fields, _) if fields.nfields() == 2 => {
                let lhs = fields.fields().next()?;
                let rhs = fields.fields().nth(1)?;
                (lhs == rhs).then(|| DType::Bool(Nullability::NonNullable))
            }
            _ => None,
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(AllIdenticalPartial {
            all_identical: true,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if !partial.all_identical {
            return Ok(());
        }
        let other_identical = other.as_bool().value().unwrap_or(false);
        if !other_identical {
            partial.all_identical = false;
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            partial.all_identical,
            Nullability::NonNullable,
        ))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.all_identical = true;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        !partial.all_identical
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !partial.all_identical {
            return Ok(());
        }

        match batch {
            Columnar::Constant(c) => {
                // A constant struct means every row is the same struct value,
                // so lhs == rhs for all rows.
                let _ = c;
                Ok(())
            }
            Columnar::Canonical(c) => {
                let Canonical::Struct(s) = c else {
                    vortex_bail!(
                        "AllIdentical expects a Struct canonical, got {:?}",
                        c.dtype()
                    );
                };

                let lhs = s.unmasked_field(0);
                let rhs = s.unmasked_field(1);

                // Compare validity masks.
                let lhs_validity = lhs.validity()?.execute_mask(lhs.len(), ctx)?;
                let rhs_validity = rhs.validity()?.execute_mask(rhs.len(), ctx)?;
                if lhs_validity != rhs_validity {
                    partial.all_identical = false;
                    return Ok(());
                }

                let valid_count = lhs_validity.true_count();
                if valid_count == 0 {
                    // Both all-null, identical.
                    return Ok(());
                }

                // Compare values per canonical type.
                let lhs_canonical = lhs.clone().execute::<Canonical>(ctx)?;
                let rhs_canonical = rhs.clone().execute::<Canonical>(ctx)?;

                partial.all_identical =
                    check_canonical_identical(&lhs_canonical, &rhs_canonical, ctx)?;

                Ok(())
            }
        }
    }

    fn finalize(&self, _partials: ArrayRef) -> VortexResult<ArrayRef> {
        vortex_bail!("AllIdentical does not support array finalization");
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            partial.all_identical,
            Nullability::NonNullable,
        ))
    }
}

/// Compare two canonical arrays for value equality (validity already checked).
fn check_canonical_identical(
    lhs: &Canonical,
    rhs: &Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match (lhs, rhs) {
        (Canonical::Null(_), Canonical::Null(_)) => Ok(true),

        (Canonical::Bool(a), Canonical::Bool(b)) => {
            use crate::arrays::bool::BoolArrayExt;
            Ok(a.to_bit_buffer() == b.to_bit_buffer())
        }

        (Canonical::Primitive(a), Canonical::Primitive(b)) => {
            match_each_native_ptype!(a.ptype(), |P| {
                Ok(a.as_slice::<P>() == b.as_slice::<P>())
            })
        }

        (Canonical::Decimal(a), Canonical::Decimal(b)) => check_decimal_identical(a, b),

        (Canonical::VarBinView(a), Canonical::VarBinView(b)) => {
            // Compare views and data buffers.
            if a.views().len() != b.views().len() {
                return Ok(false);
            }
            for (av, bv) in a.views().iter().zip(b.views().iter()) {
                if av.is_inlined() != bv.is_inlined() {
                    return Ok(false);
                }
                if av.is_inlined() {
                    if av.as_inlined() != bv.as_inlined() {
                        return Ok(false);
                    }
                } else {
                    let a_bytes = &a.buffer(av.as_view().buffer_index as usize).as_slice()
                        [av.as_view().as_range()];
                    let b_bytes = &b.buffer(bv.as_view().buffer_index as usize).as_slice()
                        [bv.as_view().as_range()];
                    if a_bytes != b_bytes {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }

        (Canonical::Struct(a), Canonical::Struct(b)) => check_struct_identical(a, b, ctx),

        (Canonical::List(a), Canonical::List(b)) => check_list_identical(a, b, ctx),

        (Canonical::FixedSizeList(a), Canonical::FixedSizeList(b)) => {
            check_fixed_size_list_identical(a, b, ctx)
        }

        (Canonical::Extension(a), Canonical::Extension(b)) => {
            all_identical(a.storage_array(), b.storage_array(), ctx)
        }

        (Canonical::Variant(_), _) | (_, Canonical::Variant(_)) => {
            vortex_bail!("Variant arrays don't support AllIdentical")
        }

        _ => Err(vortex_err!(
            "Canonical type mismatch in AllIdentical: {:?} vs {:?}",
            lhs.dtype(),
            rhs.dtype()
        )),
    }
}

#[expect(
    clippy::cognitive_complexity,
    reason = "decimal widening depends on both source value types and the chosen widest type"
)]
fn check_decimal_identical(lhs: &DecimalArray, rhs: &DecimalArray) -> VortexResult<bool> {
    if lhs.values_type() == rhs.values_type() {
        return match_each_decimal_value_type!(lhs.values_type(), |S| {
            Ok(lhs.buffer::<S>().as_ref() == rhs.buffer::<S>().as_ref())
        });
    }

    let widest = lhs.values_type().max(rhs.values_type());
    match_each_decimal_value_type!(lhs.values_type(), |L| {
        match_each_decimal_value_type!(rhs.values_type(), |R| {
            match_each_decimal_value_type!(widest, |W| {
                Ok(lhs
                    .buffer::<L>()
                    .iter()
                    .zip(rhs.buffer::<R>().iter())
                    .all(|(lhs, rhs)| {
                        <W as BigCast>::from(*lhs).vortex_expect("decimal widening should succeed")
                            == <W as BigCast>::from(*rhs)
                                .vortex_expect("decimal widening should succeed")
                    }))
            })
        })
    })
}

fn check_struct_identical(
    lhs: &StructArray,
    rhs: &StructArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if let Some((lhs, rhs)) =
        filter_valid_rows_if_needed(&lhs.clone().into_array(), &rhs.clone().into_array(), ctx)?
    {
        return all_identical(&lhs, &rhs, ctx);
    }

    for (lhs_field, rhs_field) in lhs.iter_unmasked_fields().zip(rhs.iter_unmasked_fields()) {
        if !all_identical(lhs_field, rhs_field, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}

fn check_list_identical(
    lhs: &ListViewArray,
    rhs: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if let Some((lhs, rhs)) =
        filter_valid_rows_if_needed(&lhs.clone().into_array(), &rhs.clone().into_array(), ctx)?
    {
        return all_identical(&lhs, &rhs, ctx);
    }

    if lhs.is_zero_copy_to_list() && rhs.is_zero_copy_to_list() {
        return check_zero_copy_list_identical(lhs, rhs, ctx);
    }

    let lhs = list_from_list_view(lhs.clone())?;
    let rhs = list_from_list_view(rhs.clone())?;

    if !check_list_offsets_identical(&lhs, &rhs)? {
        return Ok(false);
    }

    all_identical(lhs.elements(), rhs.elements(), ctx)
}

fn check_list_offsets_identical(lhs: &ListArray, rhs: &ListArray) -> VortexResult<bool> {
    for idx in 0..=lhs.len() {
        if lhs.offset_at(idx)? != rhs.offset_at(idx)? {
            return Ok(false);
        }
    }

    Ok(true)
}

fn check_zero_copy_list_identical(
    lhs: &ListViewArray,
    rhs: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    debug_assert!(lhs.is_zero_copy_to_list());
    debug_assert!(rhs.is_zero_copy_to_list());

    if lhs.is_empty() {
        return Ok(true);
    }

    let lhs_base = lhs.offset_at(0);
    let rhs_base = rhs.offset_at(0);

    for idx in 0..lhs.len() {
        if lhs.size_at(idx) != rhs.size_at(idx) {
            return Ok(false);
        }

        if lhs.offset_at(idx) - lhs_base != rhs.offset_at(idx) - rhs_base {
            return Ok(false);
        }
    }

    let lhs_end = lhs.offset_at(lhs.len() - 1) + lhs.size_at(lhs.len() - 1);
    let rhs_end = rhs.offset_at(rhs.len() - 1) + rhs.size_at(rhs.len() - 1);

    let lhs_elements = lhs.elements().slice(lhs_base..lhs_end)?;
    let rhs_elements = rhs.elements().slice(rhs_base..rhs_end)?;

    all_identical(&lhs_elements, &rhs_elements, ctx)
}

fn check_fixed_size_list_identical(
    lhs: &FixedSizeListArray,
    rhs: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if let Some((lhs, rhs)) =
        filter_valid_rows_if_needed(&lhs.clone().into_array(), &rhs.clone().into_array(), ctx)?
    {
        return all_identical(&lhs, &rhs, ctx);
    }

    all_identical(lhs.elements(), rhs.elements(), ctx)
}

fn filter_valid_rows_if_needed(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(ArrayRef, ArrayRef)>> {
    let validity = lhs.validity()?;
    if validity.no_nulls() {
        return Ok(None);
    }

    let mask = validity.execute_mask(lhs.len(), ctx)?;
    if mask.true_count() == lhs.len() {
        return Ok(None);
    }

    Ok(Some((lhs.filter(mask.clone())?, rhs.filter(mask)?)))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::all_identical::all_identical;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn identical_primitives() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let b = buffer![1i32, 2, 3].into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn different_primitives() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = buffer![1i32, 2, 3].into_array();
        let b = buffer![1i32, 2, 4].into_array();
        assert!(!all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_with_nulls() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let b = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn different_nulls() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let b = PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3)]).into_array();
        assert!(!all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_empty() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();
        let b = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_bools() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = BoolArray::from_iter([true, false, true]).into_array();
        let b = BoolArray::from_iter([true, false, true]).into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn different_bools() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = BoolArray::from_iter([true, false, true]).into_array();
        let b = BoolArray::from_iter([true, true, true]).into_array();
        assert!(!all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_strings() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        let b = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn different_strings() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        let b = VarBinViewArray::from_iter_str(["hello", "earth"]).into_array();
        assert!(!all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_structs() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                buffer![1i32, 2].into_array(),
                buffer![10i32, 20].into_array(),
            ],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let b = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                buffer![1i32, 2].into_array(),
                buffer![10i32, 20].into_array(),
            ],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn different_structs() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                buffer![1i32, 2].into_array(),
                buffer![10i32, 20].into_array(),
            ],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let b = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                buffer![1i32, 2].into_array(),
                buffer![10i32, 99].into_array(),
            ],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        assert!(!all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_structs_ignore_values_under_null_rows() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let validity = Validity::from_iter([true, false]);
        let a = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                buffer![1i32, 2].into_array(),
                buffer![10i32, 20].into_array(),
            ],
            2,
            validity.clone(),
        )?
        .into_array();
        let b = StructArray::try_new(
            FieldNames::from(["x", "y"]),
            vec![
                buffer![1i32, 99].into_array(),
                buffer![10i32, 999].into_array(),
            ],
            2,
            validity,
        )?
        .into_array();

        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[rstest]
    #[case(vec![1i32, 2, 3], vec![1i32, 2, 3], true)]
    #[case(vec![1i32, 2, 3], vec![1i32, 2, 4], false)]
    fn parameterized_primitive(
        #[case] a: Vec<i32>,
        #[case] b: Vec<i32>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = PrimitiveArray::from_iter(a).into_array();
        let b = PrimitiveArray::from_iter(b).into_array();
        assert_eq!(all_identical(&a, &b, &mut ctx)?, expected);
        Ok(())
    }

    #[test]
    fn identical_chunked() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let a = ChunkedArray::try_new(
            vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();
        let b = ChunkedArray::try_new(
            vec![buffer![1i32, 2].into_array(), buffer![3i32, 4].into_array()],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();
        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_lists_with_different_offset_dtypes() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let elements = buffer![1i32, 2, 3, 4].into_array();
        let a = ListViewArray::try_new(
            elements.clone(),
            buffer![0u8, 2].into_array(),
            buffer![2u8, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let b = ListViewArray::try_new(
            elements,
            buffer![0i16, 2].into_array(),
            buffer![2i16, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_decimals_with_different_value_types() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(3, 0);
        let a = DecimalArray::new(buffer![1i8, 2, 3], decimal_dtype, Validity::NonNullable)
            .into_array();
        let b = DecimalArray::new(buffer![1i16, 2, 3], decimal_dtype, Validity::NonNullable)
            .into_array();

        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_lists_ignore_null_row_garbage() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let validity = Validity::from_iter([true, false]);
        let a = ListViewArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u8, 2].into_array(),
            buffer![2u8, 2].into_array(),
            validity.clone(),
        )?
        .into_array();
        let b = ListViewArray::try_new(
            buffer![1i32, 2, 9, 8].into_array(),
            buffer![0i16, 2].into_array(),
            buffer![2i16, 2].into_array(),
            validity,
        )?
        .into_array();

        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn identical_fixed_size_lists_ignore_null_row_garbage() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let validity = Validity::from_iter([true, false]);
        let a = FixedSizeListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            2,
            validity.clone(),
            2,
        )?
        .into_array();
        let b = FixedSizeListArray::try_new(buffer![1i32, 2, 9, 8].into_array(), 2, validity, 2)?
            .into_array();

        assert!(all_identical(&a, &b, &mut ctx)?);
        Ok(())
    }
}
