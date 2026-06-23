// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod extension;
mod fixed_size_list;
mod list;
pub mod primitive;
mod struct_;
mod varbin;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use self::bool::check_bool_constant;
use self::decimal::check_decimal_constant;
use self::extension::check_extension_constant;
use self::fixed_size_list::check_fixed_size_list_constant;
use self::list::check_listview_constant;
use self::primitive::check_primitive_constant;
use self::struct_::check_struct_constant;
use self::varbin::check_varbinview_constant;
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
use crate::arrays::Constant;
use crate::arrays::Null;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Check if two arrays of the same length have equal values at every position (null-safe).
///
/// Two positions are considered equal if they are both null, or both non-null with the same value.
// TODO(ngates): move this function out when we have any/all aggregate functions.
fn arrays_value_equal(a: &ArrayRef, b: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return Ok(true);
    }

    // Check validity masks match (null positions must be identical).
    let a_mask = a.validity()?.execute_mask(a.len(), ctx)?;
    let b_mask = b.validity()?.execute_mask(b.len(), ctx)?;
    if a_mask != b_mask {
        return Ok(false);
    }

    let valid_count = a_mask.true_count();
    if valid_count == 0 {
        // Both all-null → equal.
        return Ok(true);
    }

    // Compare values element-wise. Result is null where both inputs are null,
    // true/false where both are valid.
    let eq_result = a.binary(b.clone(), Operator::Eq)?;
    let eq_result = eq_result.null_as_false().execute(ctx)?;

    Ok(eq_result.true_count() == valid_count)
}

/// Compute whether an array has constant values.
///
/// An array is constant IFF at least one of the following conditions apply:
/// 1. It has at least one element (**Note** - an empty array isn't constant).
/// 2. It's encoded as a [`ConstantArray`](crate::arrays::ConstantArray) or [`NullArray`](crate::arrays::NullArray)
/// 3. Has an exact statistic attached to it, saying its constant.
/// 4. Is all invalid.
/// 5. Is all valid AND has minimum and maximum statistics that are equal.
pub fn is_constant(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    // Short-circuit using cached array statistics.
    if let Precision::Exact(value) = array.statistics().get_as::<bool>(Stat::IsConstant) {
        return Ok(value);
    }

    // Empty arrays are not constant.
    if array.is_empty() {
        return Ok(false);
    }

    // Array of length 1 is always constant.
    if array.len() == 1 {
        array
            .statistics()
            .set(Stat::IsConstant, Precision::Exact(true.into()));
        return Ok(true);
    }

    // Constant and null arrays are always constant.
    if array.is::<Constant>() || array.is::<Null>() {
        array
            .statistics()
            .set(Stat::IsConstant, Precision::Exact(true.into()));
        return Ok(true);
    }

    let all_invalid = array.all_invalid(ctx)?;
    if all_invalid {
        array
            .statistics()
            .set(Stat::IsConstant, Precision::Exact(true.into()));
        return Ok(true);
    }

    let all_valid = array.all_valid(ctx)?;

    // If we have some nulls but not all nulls, array can't be constant.
    if !all_valid && !all_invalid {
        array
            .statistics()
            .set(Stat::IsConstant, Precision::Exact(false.into()));
        return Ok(false);
    }

    // We already know here that the array is all valid, so we check for min/max stats.
    let min_stat = array.statistics().get(Stat::Min);
    let max_stat = array.statistics().get(Stat::Max);

    if let Precision::Exact(min) = min_stat.as_ref()
        && let Precision::Exact(max) = max_stat.as_ref()
        && min == max
        && (Stat::NaNCount.dtype(array.dtype()).is_none()
            || array.statistics().get_as::<u64>(Stat::NaNCount) == Precision::exact(0u64))
    {
        array
            .statistics()
            .set(Stat::IsConstant, Precision::Exact(true.into()));
        return Ok(true);
    }

    // Short-circuit for unsupported dtypes.
    if IsConstant
        .return_dtype(&EmptyOptions, array.dtype())
        .is_none()
    {
        // Null dtype - vacuously false for empty
        return Ok(false);
    }

    // Compute using Accumulator<IsConstant>.
    let mut acc = Accumulator::try_new(IsConstant, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result_scalar = acc.finish()?;

    let result = result_scalar.as_bool().value().unwrap_or(false);

    // Cache the computed is_constant as a statistic.
    array
        .statistics()
        .set(Stat::IsConstant, Precision::Exact(result.into()));

    Ok(result)
}

/// Compute whether an array is constant.
///
/// Returns a `Bool(NonNullable)` scalar.
/// The partial state is a nullable struct `{is_constant: Bool(NN), value: input_dtype?}`.
/// A null struct means the accumulator has seen no data yet (empty).
#[derive(Clone, Debug)]
pub struct IsConstant;

impl IsConstant {
    /// Build a partial scalar from a kernel's `is_constant` result.
    ///
    /// Kernels that compute `is_constant` by delegating to child arrays can call this
    /// to package the boolean result into the partial struct format expected by the
    /// accumulator, avoiding duplicated boilerplate.
    pub fn make_partial(
        batch: &ArrayRef,
        is_constant: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let partial_dtype = make_is_constant_partial_dtype(batch.dtype());
        if is_constant {
            if batch.is_empty() {
                return Ok(Scalar::null(partial_dtype));
            }
            let first_value = batch.execute_scalar(0, ctx)?.into_nullable();
            Ok(Scalar::struct_(
                partial_dtype,
                vec![Scalar::bool(true, Nullability::NonNullable), first_value],
            ))
        } else {
            Ok(Scalar::struct_(
                partial_dtype,
                vec![
                    Scalar::bool(false, Nullability::NonNullable),
                    Scalar::null(batch.dtype().as_nullable()),
                ],
            ))
        }
    }
}

/// Partial accumulator state for is_constant.
pub struct IsConstantPartial {
    is_constant: bool,
    /// None = empty (no values seen), Some(null) = all nulls, Some(v) = first value seen.
    first_value: Option<Scalar>,
    element_dtype: DType,
}

impl IsConstantPartial {
    fn check_value(&mut self, value: Scalar) {
        if !self.is_constant {
            return;
        }
        match &self.first_value {
            None => {
                self.first_value = Some(value);
            }
            Some(first) => {
                if *first != value {
                    self.is_constant = false;
                }
            }
        }
    }
}

static NAMES: std::sync::LazyLock<FieldNames> =
    std::sync::LazyLock::new(|| FieldNames::from(["is_constant", "value"]));

pub fn make_is_constant_partial_dtype(element_dtype: &DType) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                DType::Bool(Nullability::NonNullable),
                element_dtype.as_nullable(),
            ],
        ),
        Nullability::Nullable,
    )
}

impl AggregateFnVTable for IsConstant {
    type Options = EmptyOptions;
    type Partial = IsConstantPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.is_constant")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("IsConstant is not yet serializable");
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Null | DType::Variant(..) => None,
            _ => Some(DType::Bool(Nullability::NonNullable)),
        }
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Null | DType::Variant(..) => None,
            _ => Some(make_is_constant_partial_dtype(input_dtype)),
        }
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(IsConstantPartial {
            is_constant: true,
            first_value: None,
            element_dtype: input_dtype.clone(),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if !partial.is_constant {
            return Ok(());
        }

        // Null struct means the other accumulator was empty, skip it.
        if other.is_null() {
            return Ok(());
        }

        let other_is_constant = other
            .as_struct()
            .field_by_idx(0)
            .map(|s| s.as_bool().value().unwrap_or(false))
            .unwrap_or(false);

        if !other_is_constant {
            partial.is_constant = false;
            return Ok(());
        }

        let other_value = other.as_struct().field_by_idx(1);

        if let Some(other_val) = other_value {
            partial.check_value(other_val);
        }

        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = make_is_constant_partial_dtype(&partial.element_dtype);
        Ok(match &partial.first_value {
            None => {
                // Empty accumulator — return null struct.
                Scalar::null(dtype)
            }
            Some(first_value) => Scalar::struct_(
                dtype,
                vec![
                    Scalar::bool(partial.is_constant, Nullability::NonNullable),
                    first_value
                        .clone()
                        .cast(&partial.element_dtype.as_nullable())?,
                ],
            ),
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.is_constant = true;
        partial.first_value = None;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        !partial.is_constant
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !partial.is_constant {
            return Ok(());
        }

        match batch {
            Columnar::Constant(c) => {
                partial.check_value(c.scalar().clone().into_nullable());
                Ok(())
            }
            Columnar::Canonical(c) => {
                if c.is_empty() {
                    return Ok(());
                }

                // Convert to ArrayRef for DynArrayData methods.
                let array_ref = c.clone().into_array();

                let all_invalid = array_ref.all_invalid(ctx)?;
                if all_invalid {
                    partial.check_value(Scalar::null(partial.element_dtype.as_nullable()));
                    return Ok(());
                }

                let all_valid = array_ref.all_valid(ctx)?;
                // Mixed nulls → not constant.
                if !all_valid && !all_invalid {
                    partial.is_constant = false;
                    return Ok(());
                }

                // All valid from here. Check batch-level constancy.
                if c.len() == 1 {
                    partial.check_value(array_ref.execute_scalar(0, ctx)?.into_nullable());
                    return Ok(());
                }

                let batch_is_constant = match c {
                    Canonical::Primitive(p) => check_primitive_constant(p),
                    Canonical::Bool(b) => check_bool_constant(b),
                    Canonical::VarBinView(v) => check_varbinview_constant(v),
                    Canonical::Decimal(d) => check_decimal_constant(d),
                    Canonical::Struct(s) => check_struct_constant(s, ctx)?,
                    Canonical::Extension(e) => check_extension_constant(e, ctx)?,
                    Canonical::List(l) => check_listview_constant(l, ctx)?,
                    Canonical::FixedSizeList(f) => check_fixed_size_list_constant(f, ctx)?,
                    Canonical::Null(_) => true,
                    Canonical::Variant(_) => {
                        vortex_bail!("Variant arrays don't support IsConstant")
                    }
                };

                if !batch_is_constant {
                    partial.is_constant = false;
                    return Ok(());
                }

                partial.check_value(array_ref.execute_scalar(0, ctx)?.into_nullable());
                Ok(())
            }
        }
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        partials.get_item(NAMES.get(0).vortex_expect("out of bounds").clone())
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if partial.first_value.is_none() {
            // Empty accumulator → return false.
            return Ok(Scalar::bool(false, Nullability::NonNullable));
        }
        Ok(Scalar::bool(partial.is_constant, Nullability::NonNullable))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::is_constant::is_constant;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Stat;
    use crate::validity::Validity;

    // Tests migrated from compute/is_constant.rs
    #[test]
    fn is_constant_min_max_no_nan() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let arr = buffer![0, 1].into_array();
        arr.statistics()
            .compute_all(&[Stat::Min, Stat::Max], &mut ctx)?;
        assert!(!is_constant(&arr, &mut ctx)?);

        let arr = buffer![0, 0].into_array();
        arr.statistics()
            .compute_all(&[Stat::Min, Stat::Max], &mut ctx)?;
        assert!(is_constant(&arr, &mut ctx)?);

        let arr = PrimitiveArray::from_option_iter([Some(0), Some(0)]).into_array();
        assert!(is_constant(&arr, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn is_constant_min_max_with_nan() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let arr = PrimitiveArray::from_iter([0.0, 0.0, f32::NAN]).into_array();
        arr.statistics()
            .compute_all(&[Stat::Min, Stat::Max], &mut ctx)?;
        assert!(!is_constant(&arr, &mut ctx)?);

        let arr =
            PrimitiveArray::from_option_iter([Some(f32::NEG_INFINITY), Some(f32::NEG_INFINITY)])
                .into_array();
        arr.statistics()
            .compute_all(&[Stat::Min, Stat::Max], &mut ctx)?;
        assert!(is_constant(&arr, &mut ctx)?);
        Ok(())
    }

    // Tests migrated from arrays/bool/compute/is_constant.rs
    #[rstest]
    #[case(vec![true], true)]
    #[case(vec![false; 65], true)]
    #[case({
        let mut v = vec![true; 64];
        v.push(false);
        v
    }, false)]
    fn test_bool_is_constant(#[case] input: Vec<bool>, #[case] expected: bool) -> VortexResult<()> {
        let array = BoolArray::from_iter(input);
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(is_constant(&array.into_array(), &mut ctx)?, expected);
        Ok(())
    }

    // Tests migrated from arrays/chunked/compute/is_constant.rs
    #[test]
    fn empty_chunk_is_constant() -> VortexResult<()> {
        let chunked = ChunkedArray::try_new(
            vec![
                Buffer::<u8>::empty().into_array(),
                Buffer::<u8>::empty().into_array(),
                buffer![255u8, 255].into_array(),
                Buffer::<u8>::empty().into_array(),
                buffer![255u8, 255].into_array(),
            ],
            DType::Primitive(PType::U8, Nullability::NonNullable),
        )?
        .into_array();

        let mut ctx = array_session().create_execution_ctx();
        assert!(is_constant(&chunked, &mut ctx)?);
        Ok(())
    }

    // Tests migrated from arrays/decimal/compute/is_constant.rs
    #[test]
    fn test_decimal_is_constant() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let array = DecimalArray::new(
            buffer![0i128, 1i128, 2i128],
            DecimalDType::new(19, 0),
            Validity::NonNullable,
        );
        assert!(!is_constant(&array.into_array(), &mut ctx)?);

        let array = DecimalArray::new(
            buffer![100i128, 100i128, 100i128],
            DecimalDType::new(19, 0),
            Validity::NonNullable,
        );
        assert!(is_constant(&array.into_array(), &mut ctx)?);
        Ok(())
    }

    // Tests migrated from arrays/list/compute/is_constant.rs
    #[test]
    fn test_is_constant_nested_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let xs = ListArray::try_new(
            buffer![0i32, 1, 0, 1].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?;

        let struct_of_lists = StructArray::try_new(
            FieldNames::from(["xs"]),
            vec![xs.into_array()],
            2,
            Validity::NonNullable,
        )?;
        assert!(is_constant(
            &struct_of_lists.clone().into_array(),
            &mut ctx
        )?);
        assert!(is_constant(&struct_of_lists.into_array(), &mut ctx)?);
        Ok(())
    }

    #[rstest]
    #[case(
        // [1,2], [1, 2], [1, 2]
        vec![1i32, 2, 1, 2, 1, 2],
        vec![0u32, 2, 4, 6],
        true
    )]
    #[case(
        // [1, 2], [3], [4, 5]
        vec![1i32, 2, 3, 4, 5],
        vec![0u32, 2, 3, 5],
        false
    )]
    #[case(
        // [1, 2], [3, 4]
        vec![1i32, 2, 3, 4],
        vec![0u32, 2, 4],
        false
    )]
    #[case(
        // [], [], []
        vec![],
        vec![0u32, 0, 0, 0],
        true
    )]
    fn test_list_is_constant(
        #[case] elements: Vec<i32>,
        #[case] offsets: Vec<u32>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let list_array = ListArray::try_new(
            PrimitiveArray::from_iter(elements).into_array(),
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;

        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(is_constant(&list_array.into_array(), &mut ctx)?, expected);
        Ok(())
    }

    #[test]
    fn test_list_is_constant_nested_lists() -> VortexResult<()> {
        let inner_elements = buffer![1i32, 2, 1, 2].into_array();
        let inner_offsets = buffer![0u32, 1, 2, 3, 4].into_array();
        let inner_lists = ListArray::try_new(inner_elements, inner_offsets, Validity::NonNullable)?;

        let outer_offsets = buffer![0u32, 2, 4].into_array();
        let outer_list = ListArray::try_new(
            inner_lists.into_array(),
            outer_offsets,
            Validity::NonNullable,
        )?;

        let mut ctx = array_session().create_execution_ctx();
        // Both outer lists contain [[1], [2]], so should be constant
        assert!(is_constant(&outer_list.into_array(), &mut ctx)?);
        Ok(())
    }

    #[rstest]
    #[case(
        // 100 identical [1, 2] lists
        [1i32, 2].repeat(100),
        (0..101).map(|i| (i * 2) as u32).collect(),
        true
    )]
    #[case(
        // Difference after threshold: 64 identical [1, 2] + one [3, 4]
        {
            let mut elements = [1i32, 2].repeat(64);
            elements.extend_from_slice(&[3, 4]);
            elements
        },
        (0..66).map(|i| (i * 2) as u32).collect(),
        false
    )]
    #[case(
        // Difference in first 64: first 63 identical [1, 2] + one [3, 4] + rest identical [1, 2]
        {
            let mut elements = [1i32, 2].repeat(63);
            elements.extend_from_slice(&[3, 4]);
            elements.extend([1i32, 2].repeat(37));
            elements
        },
        (0..101).map(|i| (i * 2) as u32).collect(),
        false
    )]
    fn test_list_is_constant_with_threshold(
        #[case] elements: Vec<i32>,
        #[case] offsets: Vec<u32>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let list_array = ListArray::try_new(
            PrimitiveArray::from_iter(elements).into_array(),
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::NonNullable,
        )?;

        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(is_constant(&list_array.into_array(), &mut ctx)?, expected);
        Ok(())
    }
}
