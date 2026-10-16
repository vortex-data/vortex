// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Min/max over fixed-size lists.
//!
//! Lists order lexicographically by element, exactly as `compare` and [`Scalar`] ordering do: a
//! null element orders before every non-null element, and float elements order by IEEE 754 total
//! order, so a list holding a NaN element is an ordinary value. Null lists are skipped, as for
//! every other dtype.
//!
//! [`Scalar`]: crate::scalar::Scalar

use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::scalar_fn::fns::binary::row_comparator;

pub(super) fn accumulate_fixed_size_list(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Decode the elements once, for both the comparisons and reading back the two extrema.
    let array = array
        .clone()
        .into_array()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_fixed_size_list();
    let Some((min_row, max_row)) = extreme_rows(&array, ctx)? else {
        return Ok(());
    };
    let dtype = array.dtype().as_nonnullable();
    let local = MinMaxResult {
        min: array.execute_scalar(min_row, ctx)?.cast(&dtype)?,
        max: array.execute_scalar(max_row, ctx)?.cast(&dtype)?,
    };
    partial.merge(args, Some(local));
    Ok(())
}

/// The indices of a smallest and a largest non-null list, or `None` if every list is null.
fn extreme_rows(
    array: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(usize, usize)>> {
    let valid = array.as_ref().validity()?.execute_mask(array.len(), ctx)?;
    let elements = array.elements();

    if *elements.dtype() == DType::Primitive(PType::U8, NonNullable) {
        // Lists of non-null bytes, such as UUIDs, order like byte strings.
        let list_size = array.list_size() as usize;
        let bytes = elements.clone().execute::<PrimitiveArray>(ctx)?;
        let bytes = bytes.as_slice::<u8>();
        let list = |row: usize| &bytes[row * list_size..][..list_size];
        return Ok(fold_extreme_rows(&valid, |a, b| list(a).cmp(list(b))));
    }

    Ok(fold_extreme_rows(
        &valid,
        row_comparator(array.as_ref(), ctx)?,
    ))
}

/// The first of the `valid` rows holding the smallest and the largest value under `compare`.
fn fold_extreme_rows(
    valid: &Mask,
    compare: impl Fn(usize, usize) -> Ordering,
) -> Option<(usize, usize)> {
    let all_rows = [(0, valid.len())];
    let runs = match valid.slices() {
        AllOr::All => all_rows.as_slice(),
        AllOr::None => &[],
        AllOr::Some(runs) => runs,
    };
    let mut rows = runs.iter().flat_map(|&(start, end)| start..end);

    let first = rows.next()?;
    Some(rows.fold((first, first), |(min, max), row| {
        // A row that lowers the minimum cannot also raise the maximum.
        if compare(row, min).is_lt() {
            (row, max)
        } else if compare(row, max).is_gt() {
            (min, row)
        } else {
            (min, max)
        }
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateDTypes;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::min_max::MinMaxResult;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::DictArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativePType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// A nullable fixed-size list array of `rows`, over elements of the given nullability.
    ///
    /// Null rows are backed by `T::default()` elements, which must never become an extremum.
    fn list_array<T, const N: usize>(
        rows: &[Option<[Option<T>; N]>],
        elements: Nullability,
    ) -> ArrayRef
    where
        T: NativePType,
    {
        let values = rows
            .iter()
            .flat_map(|row| row.unwrap_or([Some(T::default()); N]));
        let elements = match elements {
            NonNullable => PrimitiveArray::from_iter(
                values.map(|value| value.vortex_expect("non-nullable elements")),
            ),
            Nullable => PrimitiveArray::from_option_iter(values),
        };
        let validity = Validity::from_iter(rows.iter().map(Option::is_some));
        let list_size = u32::try_from(N).vortex_expect("list size fits in u32");
        FixedSizeListArray::try_new(elements.into_array(), list_size, validity, rows.len())
            .vortex_expect("valid fixed-size list")
            .into_array()
    }

    /// The non-nullable list scalar holding `values`.
    fn list_scalar<T, const N: usize>(values: [Option<T>; N], elements: Nullability) -> Scalar
    where
        T: NativePType,
        PValue: From<T>,
    {
        let element_dtype = DType::Primitive(T::PTYPE, elements);
        let children = values
            .into_iter()
            .map(|value| match value {
                Some(value) => Scalar::primitive(value, elements),
                None => Scalar::null(element_dtype.clone()),
            })
            .collect();
        Scalar::fixed_size_list(element_dtype, children, NonNullable)
    }

    fn min_max_of(array: &ArrayRef) -> VortexResult<Option<MinMaxResult>> {
        min_max(
            array,
            &mut SESSION.create_execution_ctx(),
            NumericalAggregateOpts::default(),
        )
    }

    fn assert_extrema<T, const N: usize>(
        array: &ArrayRef,
        elements: Nullability,
        min: [Option<T>; N],
        max: [Option<T>; N],
    ) -> VortexResult<()>
    where
        T: NativePType,
        PValue: From<T>,
    {
        assert_eq!(
            min_max_of(array)?,
            Some(MinMaxResult {
                min: list_scalar(min, elements),
                max: list_scalar(max, elements),
            })
        );
        Ok(())
    }

    #[rstest]
    #[case::non_nullable_elements(NonNullable)]
    #[case::nullable_elements(Nullable)]
    fn orders_lexicographically_not_elementwise(#[case] elements: Nullability) -> VortexResult<()> {
        // The elementwise extrema [0, 2] and [1, 10] are not lists of the array.
        let rows = [
            Some([Some(1i32), Some(5)]),
            Some([Some(0), Some(9)]),
            Some([Some(1), Some(2)]),
            Some([Some(0), Some(10)]),
        ];
        let array = list_array(&rows, elements);
        assert_extrema(&array, elements, [Some(0i32), Some(9)], [Some(1), Some(5)])
    }

    #[rstest]
    #[case::bytes(NonNullable)]
    #[case::nullable_bytes(Nullable)]
    fn skips_null_lists(#[case] elements: Nullability) -> VortexResult<()> {
        // Null lists are backed by zeroes, which would otherwise be the minimum.
        let rows = [
            None,
            Some([Some(7u8), Some(1)]),
            None,
            Some([Some(3), Some(200)]),
            Some([Some(250), Some(0)]),
            None,
        ];
        let array = list_array(&rows, elements);
        assert_extrema(
            &array,
            elements,
            [Some(3u8), Some(200)],
            [Some(250), Some(0)],
        )
    }

    #[test]
    fn null_elements_order_first() -> VortexResult<()> {
        let rows = [
            Some([Some(1i64), None]),
            Some([Some(1), Some(-5)]),
            Some([None, Some(9)]),
        ];
        let array = list_array(&rows, Nullable);
        assert_extrema(&array, Nullable, [None, Some(9i64)], [Some(1), Some(-5)])
    }

    /// Float elements order by IEEE 754 total order: -NaN first, NaN last, and -0.0 before 0.0.
    #[rstest]
    #[case::nans(
        &[0.0, f64::NAN, -0.0, -f64::NAN, f64::INFINITY],
        -f64::NAN,
        f64::NAN
    )]
    #[case::signed_zeros(&[0.0, -0.0, 0.0], -0.0, 0.0)]
    fn float_elements_order_by_total_order(
        #[case] values: &[f64],
        #[case] min: f64,
        #[case] max: f64,
    ) -> VortexResult<()> {
        let rows = values.iter().map(|&v| Some([Some(v)])).collect::<Vec<_>>();
        let result =
            min_max_of(&list_array(&rows, NonNullable))?.vortex_expect("lists have extrema");
        let bits = |list: &Scalar| {
            f64::try_from(&list.as_list().element(0).vortex_expect("one element")).map(f64::to_bits)
        };
        assert_eq!(bits(&result.min)?, min.to_bits());
        assert_eq!(bits(&result.max)?, max.to_bits());
        Ok(())
    }

    #[rstest]
    #[case::bytes(NonNullable)]
    #[case::nullable_bytes(Nullable)]
    fn degenerate_lists_are_equal(#[case] elements: Nullability) -> VortexResult<()> {
        let array = list_array::<u8, 0>(&[Some([]), None, Some([])], elements);
        assert_extrema::<u8, 0>(&array, elements, [], [])
    }

    #[rstest]
    #[case::bytes(NonNullable)]
    #[case::nullable_bytes(Nullable)]
    fn sliced_lists(#[case] elements: Nullability) -> VortexResult<()> {
        // The slice drops the overall extrema in its first and last rows.
        let rows = [
            Some([Some(0u8), Some(0)]),
            Some([Some(4), Some(1)]),
            None,
            Some([Some(2), Some(9)]),
            Some([Some(9), Some(9)]),
        ];
        let array = list_array(&rows, elements).slice(1..4)?;
        assert_extrema(&array, elements, [Some(2u8), Some(9)], [Some(4), Some(1)])
    }

    #[test]
    fn dictionary_values_count_only_when_referenced() -> VortexResult<()> {
        // The unreferenced value 0 would otherwise be the minimum.
        let values = list_array(
            &[
                Some([Some(0u8), Some(0)]),
                Some([Some(5), Some(5)]),
                Some([Some(7), Some(1)]),
            ],
            NonNullable,
        );
        let array = DictArray::new(buffer![1u32, 2, 1].into_array(), values).into_array();
        assert_extrema(
            &array,
            NonNullable,
            [Some(5u8), Some(5)],
            [Some(7), Some(1)],
        )
    }

    #[test]
    fn bool_elements() -> VortexResult<()> {
        let elements = BoolArray::new(
            BitBuffer::from([true, false, false, true, true, true, false, false].as_slice()),
            Validity::NonNullable,
        );
        let array =
            FixedSizeListArray::try_new(elements.into_array(), 2, Validity::NonNullable, 4)?
                .into_array();
        assert_extrema_rows(&array, 3, 2)
    }

    #[test]
    fn decimal_elements() -> VortexResult<()> {
        let elements = DecimalArray::new(
            buffer![150i32, -20, 150, -30, -999, 999, 150, 0],
            DecimalDType::new(4, 2),
            Validity::NonNullable,
        );
        let array =
            FixedSizeListArray::try_new(elements.into_array(), 2, Validity::NonNullable, 4)?
                .into_array();
        assert_extrema_rows(&array, 2, 3)
    }

    /// Assert that the extrema of `array` are its rows `min` and `max`.
    fn assert_extrema_rows(array: &ArrayRef, min: usize, max: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max_of(array)?,
            Some(MinMaxResult {
                min: array.execute_scalar(min, &mut ctx)?,
                max: array.execute_scalar(max, &mut ctx)?,
            })
        );
        Ok(())
    }

    #[test]
    fn non_nullable_byte_lists() -> VortexResult<()> {
        // Sixteen-byte lists, laid out like UUIDs.
        let mut bytes = [0u8; 3 * 16];
        bytes[0] = 0x55;
        bytes[16..32].fill(0xff);
        bytes[47] = 0x01;
        let array = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(bytes).into_array(),
            16,
            Validity::NonNullable,
            3,
        )?
        .into_array();
        assert_extrema_rows(&array, 2, 1)
    }

    #[rstest]
    #[case::empty(list_array::<u8, 2>(&[], NonNullable))]
    #[case::all_null(list_array::<u8, 2>(&[None, None], NonNullable))]
    #[case::all_null_nullable_elements(list_array::<i32, 2>(&[None], Nullable))]
    fn no_extrema(#[case] array: ArrayRef) -> VortexResult<()> {
        assert_eq!(min_max_of(&array)?, None);
        Ok(())
    }

    /// Wide lists such as embedding vectors get extrema like any other list, in every encoding.
    /// Keeping stored statistics small is up to the writers that store them.
    #[test]
    fn wide_lists_get_extrema_in_every_encoding() -> VortexResult<()> {
        const WIDTH: usize = 768;
        let mut ctx = SESSION.create_execution_ctx();
        let low = list_scalar([Some(1.0f32); WIDTH], NonNullable);
        let high = list_scalar([Some(2.0f32); WIDTH], NonNullable);
        let lists = list_array::<f32, WIDTH>(
            &[Some([Some(2.0); WIDTH]), Some([Some(1.0); WIDTH])],
            NonNullable,
        );
        let constant = ConstantArray::new(high.clone(), 3).into_array();
        let dict = DictArray::new(buffer![1u32, 0].into_array(), lists.clone()).into_array();

        for (array, min) in [(lists, &low), (dict, &low), (constant, &high)] {
            let mut acc = Accumulator::try_new(
                MinMax,
                NumericalAggregateOpts::default(),
                array.dtype().clone(),
            )?;
            acc.accumulate(&array, &mut ctx)?;
            let extrema = MinMaxResult::from_scalar(&acc.finish()?)?;
            assert_eq!(
                extrema,
                Some(MinMaxResult {
                    min: min.clone(),
                    max: high.clone(),
                }),
                "extrema for {}",
                array.encoding_id()
            );
        }
        Ok(())
    }

    #[test]
    fn constant_lists() -> VortexResult<()> {
        let list = list_scalar([Some(4u16), Some(2)], NonNullable);
        let array = ConstantArray::new(list.clone(), 3).into_array();
        assert_eq!(
            min_max_of(&array)?,
            Some(MinMaxResult {
                min: list.clone(),
                max: list,
            })
        );
        Ok(())
    }

    #[test]
    fn chunks_merge_into_one_result() -> VortexResult<()> {
        let chunks = vec![
            list_array(&[Some([Some(5u8), Some(5)]), None], NonNullable),
            list_array::<u8, 2>(&[None], NonNullable),
            list_array(
                &[Some([Some(9u8), Some(0)]), Some([Some(2), Some(7)])],
                NonNullable,
            ),
        ];
        let dtype = chunks[0].dtype().clone();
        let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        assert_extrema(
            &array,
            NonNullable,
            [Some(2u8), Some(7)],
            [Some(9), Some(0)],
        )
    }

    #[test]
    fn accumulators_merge_batches_and_partials() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let batch = |rows: &[Option<[Option<i32>; 2]>]| list_array(rows, Nullable);
        let dtype = batch(&[]).dtype().clone();
        let options = NumericalAggregateOpts::default();

        let mut first = Accumulator::try_new(MinMax, options, dtype.clone())?;
        first.accumulate(&batch(&[Some([Some(3), None]), None]), &mut ctx)?;
        first.accumulate(&batch(&[Some([Some(3), Some(1)])]), &mut ctx)?;
        let mut second = Accumulator::try_new(MinMax, options, dtype.clone())?;
        second.accumulate(&batch(&[Some([Some(-4), Some(8)])]), &mut ctx)?;

        let owned = AggregateDTypes::try_new(&MinMax, &options, dtype)?;
        let args = owned.args(&options);
        let merged = MinMax.merge_partials(
            args,
            MinMax.partial_from_scalar(args, &first.finish()?)?,
            MinMax.partial_from_scalar(args, &second.finish()?)?,
        )?;
        assert_eq!(
            MinMaxResult::from_scalar(&MinMax.to_scalar(args, &merged)?)?,
            Some(MinMaxResult {
                min: list_scalar([Some(-4), Some(8)], Nullable),
                max: list_scalar([Some(3), Some(1)], Nullable),
            })
        );
        Ok(())
    }

    /// The byte fast path agrees with the general path, which the same bytes take when their
    /// element dtype is nullable.
    #[test]
    fn byte_fast_path_matches_general_path() -> VortexResult<()> {
        let byte_lists = DType::FixedSizeList(
            DType::Primitive(PType::U8, NonNullable).into(),
            3,
            NonNullable,
        );
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..200 {
            let rows = (0..rng.random_range(0..32))
                .map(|_| {
                    (!rng.random_bool(0.2))
                        .then(|| [(); 3].map(|()| Some(rng.random_range(0..4u8) * 85)))
                })
                .collect::<Vec<_>>();

            let fast = min_max_of(&list_array(&rows, NonNullable))?;
            let general = min_max_of(&list_array(&rows, Nullable))?
                .map(|MinMaxResult { min, max }| {
                    VortexResult::Ok(MinMaxResult {
                        min: min.cast(&byte_lists)?,
                        max: max.cast(&byte_lists)?,
                    })
                })
                .transpose()?;
            assert_eq!(fast, general, "extrema of {rows:?}");
        }
        Ok(())
    }

    /// The extrema are the ones [`Scalar`] ordering picks, and `compare` places every non-null
    /// list between them, which is what makes pruning against them sound.
    #[test]
    fn extrema_agree_with_scalar_ordering_and_compare() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..300 {
            let elements = if rng.random_bool(0.5) {
                Nullable
            } else {
                NonNullable
            };
            let rows = (0..rng.random_range(1..16))
                .map(|_| {
                    (!rng.random_bool(0.2)).then(|| {
                        [(); 2].map(|()| {
                            (elements == NonNullable || !rng.random_bool(0.25))
                                .then(|| rng.random_range(-1..2i16))
                        })
                    })
                })
                .collect::<Vec<_>>();
            let array = list_array(&rows, elements);
            let valid_rows = (0..rows.len()).filter(|&row| rows[row].is_some());

            let Some(result) = min_max_of(&array)? else {
                assert_eq!(valid_rows.count(), 0, "no extrema for {rows:?}");
                continue;
            };

            let lists = valid_rows
                .clone()
                .map(|row| array.execute_scalar(row, &mut ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            let scalar_min = lists.iter().min_by(|a, b| a.partial_cmp(b).unwrap());
            let scalar_max = lists.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(Some(&result.min), scalar_min, "min of {rows:?}");
            assert_eq!(Some(&result.max), scalar_max, "max of {rows:?}");

            for (bound, op) in [(&result.min, Operator::Gte), (&result.max, Operator::Lte)] {
                let bound_array = ConstantArray::new(bound.clone(), array.len()).into_array();
                let within = array
                    .binary(bound_array, op)?
                    .execute::<BoolArray>(&mut ctx)?
                    .into_bit_buffer();
                for row in valid_rows.clone() {
                    assert!(within.value(row), "row {row} of {rows:?} not {op} {bound}");
                }
            }
        }
        Ok(())
    }
}
