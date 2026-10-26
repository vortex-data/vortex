// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Demand masks for aggregating the children of nested arrays.
//!
//! A nested array stores its children unmasked: a struct field holds a value in every row, including
//! the rows where the struct is null, and a list's elements may include values that only null lists
//! reference. Aggregating such a child directly folds in values the logical column does not contain.
//!
//! A *demand* mask is false in the rows a parent nulls out: those rows are null in the logical
//! column, whatever the child holds there. The helpers here derive the demand on a child from the
//! demand on its parent, so that the child can be aggregated with
//! [`DynAccumulator::accumulate_demanded`] as its logical column, without first being masked:
//!
//! - a struct field is demanded in the rows where the struct is demanded and valid, see
//!   [`struct_field_demand`];
//! - a list element belongs to the logical elements column only if a demanded, valid list
//!   references it. Elements no such list references are not null but absent, so they cannot be
//!   described by a demand mask: a writer drops them instead, keeping only the elements that
//!   [`list_elements_demand`] selects, so that the remaining elements are all demanded.
//!
//! [`DynAccumulator::accumulate_demanded`]: crate::aggregate_fn::DynAccumulator::accumulate_demanded

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

/// Intersect `demand` with `validity`: the demanded rows that hold a value.
pub fn valid_demand(
    validity: &Validity,
    demand: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    if validity.definitely_no_nulls() || demand.all_false() {
        return Ok(demand.clone());
    }
    let valid = validity.execute_mask(demand.len(), ctx)?;
    Ok(&valid & demand)
}

/// The demand on each field of `array`: the rows demanded by `demand` where the struct is valid.
pub fn struct_field_demand(
    array: &StructArray,
    demand: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    vortex_ensure_eq!(
        demand.len(),
        array.len(),
        "Demand mask length must match the struct length",
    );
    valid_demand(&array.validity()?, demand, ctx)
}

/// The elements of `array` that belong to its logical elements column: the elements of each list
/// that `demand` selects and that is valid.
///
/// Elements outside such lists, including the elements of null lists and any elements before the
/// first or after the last offset, are not part of the elements column.
pub fn list_elements_demand(
    array: &ListArray,
    demand: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    vortex_ensure_eq!(
        demand.len(),
        array.len(),
        "Demand mask length must match the list length",
    );
    let elements_len = array.elements().len();
    let lists = valid_demand(&array.validity()?, demand, ctx)?;
    if lists.all_false() || elements_len == 0 {
        return Ok(Mask::new_false(elements_len));
    }

    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    // Offsets are non-negative, so they can be read as unsigned.
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());

    let mut buffer = BitBufferMut::new_unset(elements_len);
    match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        let offsets = offsets.as_slice::<O>();
        // Offsets are monotonic, so a run of demanded lists covers one contiguous element range.
        for_each_demanded_run(&lists, |start, end| {
            fill_elements(&mut buffer, offsets[start].as_(), offsets[end].as_());
        });
    });
    Ok(Mask::from_buffer(buffer.freeze()))
}

fn fill_elements(buffer: &mut BitBufferMut, start: usize, end: usize) {
    if start < end {
        buffer.fill_range(start, end, true);
    }
}

/// Call `run` with the `start..end` of every run of consecutive lists selected by `lists`.
fn for_each_demanded_run(lists: &Mask, mut run: impl FnMut(usize, usize)) {
    match lists.slices() {
        AllOr::All => run(0, lists.len()),
        AllOr::None => {}
        AllOr::Some(slices) => {
            for &(start, end) in slices {
                run(start, end);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::LazyLock;

    use rand::prelude::*;
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::SessionExt;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::combined::PairOptions;
    use crate::aggregate_fn::fns::all_nan::AllNan;
    use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
    use crate::aggregate_fn::fns::all_non_null::AllNonNull;
    use crate::aggregate_fn::fns::all_null::AllNull;
    use crate::aggregate_fn::fns::bounded_max::BoundedMax;
    use crate::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
    use crate::aggregate_fn::fns::bounded_min::BoundedMin;
    use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
    use crate::aggregate_fn::fns::count::Count;
    use crate::aggregate_fn::fns::first::First;
    use crate::aggregate_fn::fns::is_constant::IsConstant;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::last::Last;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::mean::Mean;
    use crate::aggregate_fn::fns::min::Min;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::nan_count::NanCount;
    use crate::aggregate_fn::fns::null_count::NullCount;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum_v2::SumV2;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::aggregate_fn::session::AggregateFnSession;
    use crate::array::VTable;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::VarBinViewArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn ctx() -> ExecutionCtx {
        SESSION.create_execution_ctx()
    }

    fn numeric_aggregates() -> Vec<AggregateFnRef> {
        let opts = NumericalAggregateOpts::default();
        vec![
            Sum.bind(opts),
            SumV2.bind(opts),
            Min.bind(opts),
            Max.bind(opts),
            MinMax.bind(opts),
            Count.bind(opts),
            Count.bind(NumericalAggregateOpts::include_nans()),
            Mean::combined().bind(PairOptions(opts, opts)),
            NullCount.bind(EmptyOptions),
            AllNull.bind(EmptyOptions),
            AllNonNull.bind(EmptyOptions),
            First.bind(EmptyOptions),
            Last.bind(EmptyOptions),
            IsConstant.bind(EmptyOptions),
            IsSorted.bind(IsSortedOptions { strict: false }),
            IsSorted.bind(IsSortedOptions { strict: true }),
            UncompressedSizeInBytes.bind(EmptyOptions),
        ]
    }

    fn float_aggregates() -> Vec<AggregateFnRef> {
        let mut aggregates = numeric_aggregates();
        aggregates.extend([
            NanCount.bind(EmptyOptions),
            AllNan.bind(EmptyOptions),
            AllNonNan.bind(EmptyOptions),
            Sum.bind(NumericalAggregateOpts::include_nans()),
            Min.bind(NumericalAggregateOpts::include_nans()),
        ]);
        aggregates
    }

    fn string_aggregates() -> Vec<AggregateFnRef> {
        let max_bytes = NonZeroUsize::new(2).expect("non zero");
        vec![
            Min.bind(NumericalAggregateOpts::default()),
            Max.bind(NumericalAggregateOpts::default()),
            BoundedMin.bind(BoundedMinOptions { max_bytes }),
            BoundedMax.bind(BoundedMaxOptions { max_bytes }),
            NullCount.bind(EmptyOptions),
            First.bind(EmptyOptions),
            IsConstant.bind(EmptyOptions),
        ]
    }

    /// Accumulating a demanded batch gives the same partial as accumulating the batch masked to null
    /// outside the demand.
    fn assert_demand_matches_mask(
        aggregates: &[AggregateFnRef],
        batch: &ArrayRef,
        demand: &Mask,
    ) -> VortexResult<()> {
        let mut ctx = ctx();
        for aggregate_fn in aggregates {
            let mut demanded = aggregate_fn.accumulator(batch.dtype())?;
            demanded.accumulate_demanded(batch, demand, &mut ctx)?;

            let masked = batch.clone().mask(demand.clone().into_array())?;
            let mut logical = aggregate_fn.accumulator(masked.dtype())?;
            logical.accumulate(&masked, &mut ctx)?;

            assert_eq!(
                demanded.flush()?,
                logical.flush()?,
                "{aggregate_fn} over {} demanded by {demand:?}",
                batch.display_values(&SESSION.create_execution_ctx()),
            );
        }
        Ok(())
    }

    fn demands(len: usize) -> Vec<Mask> {
        vec![
            Mask::new_true(len),
            Mask::new_false(len),
            Mask::from_iter((0..len).map(|i| i % 2 == 0)),
            Mask::from_iter((0..len).map(|i| i % 3 != 1)),
            Mask::from_indices(len, [len - 1]),
        ]
    }

    #[rstest]
    #[case::non_nullable(buffer![5i32, -9, 3, 3, 7, 1].into_array())]
    #[case::nullable(
        PrimitiveArray::from_option_iter([Some(5i64), None, Some(3), Some(3), None, Some(1)])
            .into_array()
    )]
    #[case::sorted_where_demanded(buffer![1u8, 9, 2, 0, 3, 0].into_array())]
    #[case::constant(ConstantArray::new(4i32, 6).into_array())]
    #[case::dict(
        DictArray::new(buffer![0u32, 1, 0, 2, 2, 1].into_array(), buffer![7i32, -2, 4].into_array())
            .into_array()
    )]
    #[case::bool(BoolArray::from_iter([true, false, true, true, false, false]).into_array())]
    fn demand_matches_mask(#[case] batch: ArrayRef) -> VortexResult<()> {
        let aggregates: Vec<_> = numeric_aggregates()
            .into_iter()
            .filter(|aggregate_fn| aggregate_fn.return_dtype(batch.dtype()).is_some())
            .collect();
        for demand in demands(batch.len()) {
            assert_demand_matches_mask(&aggregates, &batch, &demand)?;
        }
        Ok(())
    }

    #[test]
    fn demand_matches_mask_over_floats() -> VortexResult<()> {
        let batch = PrimitiveArray::from_option_iter([
            Some(f64::NAN),
            Some(2.5),
            None,
            Some(f64::NAN),
            Some(-1.0),
            None,
        ])
        .into_array();
        for demand in demands(batch.len()) {
            assert_demand_matches_mask(&float_aggregates(), &batch, &demand)?;
        }
        Ok(())
    }

    #[test]
    fn demand_matches_mask_over_strings() -> VortexResult<()> {
        let batch = VarBinViewArray::from_iter_nullable_str([
            Some("zebra"),
            Some("apple"),
            None,
            Some("mango"),
            Some("apple"),
        ])
        .into_array();
        for demand in demands(batch.len()) {
            assert_demand_matches_mask(&string_aggregates(), &batch, &demand)?;
        }
        Ok(())
    }

    #[test]
    fn demand_length_must_match_batch() -> VortexResult<()> {
        let batch = buffer![1i32, 2, 3].into_array();
        let mut acc = Sum
            .bind(NumericalAggregateOpts::default())
            .accumulator(batch.dtype())?;
        assert!(
            acc.accumulate_demanded(&batch, &Mask::new_true(2), &mut ctx())
                .is_err()
        );
        Ok(())
    }

    /// Fully demanded batches take the plain path, which caches the result on the batch; partially
    /// demanded batches never cache a statistic on the batch.
    #[test]
    fn only_fully_demanded_batches_cache_statistics() -> VortexResult<()> {
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;
        use crate::expr::stats::StatsProviderExt;

        let mut ctx = ctx();
        let min = Min.bind(NumericalAggregateOpts::default());

        let partially = buffer![5i32, 1, 3].into_array();
        let mut acc = min.accumulator(partially.dtype())?;
        acc.accumulate_demanded(&partially, &Mask::from_iter([true, false, true]), &mut ctx)?;
        assert_eq!(
            acc.finish()?,
            Scalar::primitive(3i32, Nullability::Nullable)
        );
        assert_eq!(
            partially.statistics().get_as::<i32>(Stat::Min),
            Precision::Absent
        );

        let fully = buffer![5i32, 1, 3].into_array();
        acc.accumulate_demanded(&fully, &Mask::new_true(3), &mut ctx)?;
        assert_eq!(
            acc.finish()?,
            Scalar::primitive(1i32, Nullability::Nullable)
        );
        assert_eq!(
            fully.statistics().get_as::<i32>(Stat::Min),
            Precision::exact(1i32)
        );
        Ok(())
    }

    /// A sentinel sum, distinguishable from any sum the test batches produce.
    #[derive(Debug)]
    struct SentinelSumKernel;
    impl DynAggregateKernel for SentinelSumKernel {
        fn aggregate(
            &self,
            _aggregate_fn: &AggregateFnRef,
            _batch: &ArrayRef,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<Scalar>> {
            Ok(Some(Scalar::primitive(42i64, Nullability::Nullable)))
        }
    }

    /// The demanded rows are filtered lazily, and a dictionary filters only its codes, so the
    /// dictionary's kernel still fires.
    #[test]
    fn demand_keeps_encoding_kernels() -> VortexResult<()> {
        static KERNEL: SentinelSumKernel = SentinelSumKernel;
        let session = VortexSession::empty();
        session
            .get::<AggregateFnSession>()
            .register_aggregate_kernel(Dict.id(), Some(Sum.id()), &KERNEL);
        let mut ctx = session.create_execution_ctx();

        let batch = DictArray::try_new(
            buffer![0u32, 1, 0].into_array(),
            buffer![7i32, 9].into_array(),
        )?
        .into_array();
        let mut acc = Sum
            .bind(NumericalAggregateOpts::default())
            .accumulator(batch.dtype())?;
        acc.accumulate_demanded(&batch, &Mask::from_iter([true, false, true]), &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(42i64, Nullability::Nullable)
        );
        Ok(())
    }

    /// Aggregating a struct field under the field demand matches aggregating the logical field,
    /// whose values are null wherever the struct is null.
    #[test]
    fn struct_field_aggregates_ignore_values_under_null_parents() -> VortexResult<()> {
        let mut ctx = ctx();
        // The field holds 100 and -100 under the struct's null rows.
        let field =
            PrimitiveArray::from_option_iter([Some(1i32), Some(100), None, Some(3), Some(-100)])
                .into_array();
        let array = StructArray::try_from_iter_with_validity(
            [("a", field.clone())],
            Validity::from_iter([true, false, true, true, false]),
        )?;
        let logical_field = array.clone().into_array().get_item("a")?;

        let field_demand = struct_field_demand(&array, &Mask::new_true(array.len()), &mut ctx)?;
        let aggregates = [
            Min.bind(NumericalAggregateOpts::default()),
            Max.bind(NumericalAggregateOpts::default()),
            Sum.bind(NumericalAggregateOpts::default()),
            NullCount.bind(EmptyOptions),
            IsConstant.bind(EmptyOptions),
        ];
        for aggregate_fn in &aggregates {
            let mut demanded = aggregate_fn.accumulator(field.dtype())?;
            demanded.accumulate_demanded(&field, &field_demand, &mut ctx)?;
            let mut logical = aggregate_fn.accumulator(logical_field.dtype())?;
            logical.accumulate(&logical_field, &mut ctx)?;
            assert_eq!(demanded.finish()?, logical.finish()?, "{aggregate_fn}");
        }

        let mut min = aggregates[0].accumulator(field.dtype())?;
        min.accumulate_demanded(&field, &field_demand, &mut ctx)?;
        assert_eq!(
            min.finish()?,
            Scalar::primitive(1i32, Nullability::Nullable)
        );
        // The field's own null, and the two rows the struct nulls out.
        let mut null_count = aggregates[3].accumulator(field.dtype())?;
        null_count.accumulate_demanded(&field, &field_demand, &mut ctx)?;
        assert_eq!(
            null_count.finish()?,
            Scalar::primitive(3u64, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn non_nullable_parents_demand_everything() -> VortexResult<()> {
        let mut ctx = ctx();
        let array = StructArray::try_from_iter([("a", buffer![1i32, 2].into_array())])?;
        assert!(struct_field_demand(&array, &Mask::new_true(2), &mut ctx)?.all_true());
        Ok(())
    }

    #[test]
    fn struct_field_demand_intersects_validity() -> VortexResult<()> {
        let array = StructArray::try_from_iter_with_validity(
            [("a", buffer![1i32, 2, 3, 4].into_array())],
            Validity::from_iter([true, false, true, true]),
        )?;
        let demand = Mask::from_iter([true, true, false, true]);

        let field_demand = struct_field_demand(&array, &demand, &mut ctx())?;

        assert_eq!(field_demand, Mask::from_iter([true, false, false, true]));
        Ok(())
    }

    #[test]
    fn list_elements_demand_skips_null_lists_and_unreferenced_elements() -> VortexResult<()> {
        // Lists [1, 2], [3] (null) and [4, 5] over elements that start with an unreferenced 0.
        let array = ListArray::try_new(
            buffer![0i32, 1, 2, 3, 4, 5, 6].into_array(),
            buffer![1u16, 3, 4, 6].into_array(),
            Validity::from_iter([true, false, true]),
            &mut SESSION.create_execution_ctx(),
        )?;

        let elements_demand = list_elements_demand(&array, &Mask::new_true(3), &mut ctx())?;

        assert_eq!(
            elements_demand,
            Mask::from_iter([false, true, true, false, true, true, false])
        );
        Ok(())
    }

    /// The elements of the demanded, valid lists, computed one element at a time.
    fn reference_elements_demand(ranges: &[(u32, u32)], lists: &Mask, elements_end: u32) -> Mask {
        Mask::from_iter((0..elements_end).map(|element| {
            ranges
                .iter()
                .enumerate()
                .any(|(row, &(start, end))| lists.value(row) && start <= element && element < end)
        }))
    }

    /// Run-based list element demand agrees with the element-by-element reference, across empty
    /// and null lists and partial demand.
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    fn elements_demand_matches_reference(#[case] seed: u64) -> VortexResult<()> {
        let mut ctx = ctx();
        let mut rng = StdRng::seed_from_u64(seed);
        let len = 64;
        let mut offsets = vec![0u32];
        for _ in 0..len {
            let last = *offsets.last().expect("non-empty");
            offsets.push(last + rng.random_range(0..4u32));
        }
        let elements_end = *offsets.last().expect("non-empty");
        let ranges: Vec<(u32, u32)> = offsets.windows(2).map(|w| (w[0], w[1])).collect();
        let valid = Mask::from_iter((0..len).map(|_| rng.random_bool(0.7)));
        let demand = Mask::from_iter((0..len).map(|_| rng.random_bool(0.8)));
        let expected = reference_elements_demand(&ranges, &(&valid & &demand), elements_end);
        let elements: ArrayRef = (0..elements_end).collect::<Buffer<u32>>().into_array();
        let validity = Validity::from_mask(valid, Nullability::Nullable);

        let list = ListArray::try_new(
            elements,
            offsets
                .iter()
                .copied()
                .collect::<Buffer<u32>>()
                .into_array(),
            validity,
            &mut ctx,
        )?;
        assert_eq!(list_elements_demand(&list, &demand, &mut ctx)?, expected);
        Ok(())
    }
}
