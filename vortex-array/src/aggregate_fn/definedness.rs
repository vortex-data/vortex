// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definedness: which rows of an array belong to the logical column an aggregate describes.
//!
//! A nested array stores its children unmasked: a struct field holds a value in every row, including
//! the rows where the struct is null. Aggregating such a child directly folds in values the logical
//! column does not contain.
//!
//! *Definedness* is a [`Mask`] with one bit per row of a batch. A defined row belongs to the logical
//! column with the value the batch holds there, which may itself be null. An undefined row is
//! null in the logical column, because an ancestor nulls it out, whatever the batch holds there.
//! A batch is aggregated under a definedness mask with [`DynAccumulator::accumulate_defined`].
//!
//! # Semantics
//!
//! An aggregate gives definedness meaning through [`AggregateFnVTable::try_accumulate_defined`]:
//!
//! - By default, undefined rows read as null: the result equals aggregating the batch masked by
//!   the definedness. This suits every aggregate of the logical values, such as a minimum, a sum or a
//!   null count. The masked batch has the input dtype made nullable, so an aggregate that keeps
//!   the default must have the same partial dtype for a nullable input as for the non-nullable
//!   input.
//! - An aggregate that describes the stored array rather than its logical values defines its own
//!   semantics. [`UncompressedSizeInBytes`] measures the bytes the batch holds, which definedness
//!   does not change, so it ignores the definedness.
//!
//! # Deriving definedness
//!
//! The helpers here derive the definedness on a child from the definedness on its parent:
//!
//! - a struct field is defined in the rows where the struct is defined and valid, see
//!   [`struct_field_definedness`];
//! - a list element belongs to the logical elements column only if a defined, valid list
//!   references it. Elements no such list references are not null but absent, so they cannot be
//!   described by definedness: a writer drops them instead, keeping only the elements that
//!   [`list_elements_definedness`] selects, so that the remaining elements are all defined.
//!
//! [`DynAccumulator::accumulate_defined`]: crate::aggregate_fn::DynAccumulator::accumulate_defined
//! [`AggregateFnVTable::try_accumulate_defined`]: crate::aggregate_fn::AggregateFnVTable::try_accumulate_defined
//! [`UncompressedSizeInBytes`]: crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes

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

/// Intersect `defined` with `validity`: the defined rows that hold a value.
pub fn valid_definedness(
    validity: &Validity,
    defined: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    if validity.definitely_no_nulls() || defined.all_false() {
        return Ok(defined.clone());
    }
    let valid = validity.execute_mask(defined.len(), ctx)?;
    Ok(&valid & defined)
}

/// The definedness on each field of `array`: the rows defined by `defined` where the struct is valid.
pub fn struct_field_definedness(
    array: &StructArray,
    defined: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    vortex_ensure_eq!(
        defined.len(),
        array.len(),
        "Definedness mask length must match the struct length",
    );
    valid_definedness(&array.validity()?, defined, ctx)
}

/// The elements of `array` that belong to its logical elements column: the elements of each list
/// that `defined` selects and that is valid.
///
/// Elements outside such lists, including the elements of null lists and any elements before the
/// first or after the last offset, are not part of the elements column.
pub fn list_elements_definedness(
    array: &ListArray,
    defined: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Mask> {
    vortex_ensure_eq!(
        defined.len(),
        array.len(),
        "Definedness mask length must match the list length",
    );
    let elements_len = array.elements().len();
    let lists = valid_definedness(&array.validity()?, defined, ctx)?;
    if lists.all_false() || elements_len == 0 {
        return Ok(Mask::new_false(elements_len));
    }

    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    // Offsets are non-negative, so they can be read as unsigned.
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());

    let mut buffer = BitBufferMut::new_unset(elements_len);
    match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        let offsets = offsets.as_slice::<O>();
        // Offsets are monotonic, so a run of defined lists covers one contiguous element range.
        for_each_defined_run(&lists, |start, end| {
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
fn for_each_defined_run(lists: &Mask, mut run: impl FnMut(usize, usize)) {
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
    use crate::aggregate_fn::AggregateDTypes;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::combined::PairOptions;
    use crate::aggregate_fn::fns::all_nan::AllNan;
    use crate::aggregate_fn::fns::all_non_distinct::AllNonDistinct;
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
    use crate::arrays::DecimalArray;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::VarBinViewArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn ctx() -> ExecutionCtx {
        SESSION.create_execution_ctx()
    }

    /// Aggregates that keep the default definedness semantics. `UncompressedSizeInBytes` defines its own,
    /// see `uncompressed_size_ignores_definedness`.
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

    /// Accumulating a defined batch gives the same partial as accumulating the batch masked to null
    /// outside the definedness.
    fn assert_definedness_matches_mask(
        aggregates: &[AggregateFnRef],
        batch: &ArrayRef,
        defined: &Mask,
    ) -> VortexResult<()> {
        let mut ctx = ctx();
        for aggregate_fn in aggregates {
            let mut defined_acc = aggregate_fn.accumulator(batch.dtype())?;
            defined_acc.accumulate_defined(batch, defined, &mut ctx)?;

            let masked = batch.clone().mask(defined.clone().into_array())?;
            let mut logical = aggregate_fn.accumulator(masked.dtype())?;
            logical.accumulate(&masked, &mut ctx)?;

            assert_eq!(
                defined_acc.flush()?,
                logical.flush()?,
                "{aggregate_fn} over {} defined by {defined:?}",
                batch.display_values(&SESSION.create_execution_ctx()),
            );
        }
        Ok(())
    }

    fn defined_masks(len: usize) -> Vec<Mask> {
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
    #[case::sorted_where_defined(buffer![1u8, 9, 2, 0, 3, 0].into_array())]
    #[case::constant(ConstantArray::new(4i32, 6).into_array())]
    #[case::dict(
        DictArray::new(buffer![0u32, 1, 0, 2, 2, 1].into_array(), buffer![7i32, -2, 4].into_array())
            .into_array()
    )]
    #[case::bool(BoolArray::from_iter([true, false, true, true, false, false]).into_array())]
    fn definedness_matches_mask(#[case] batch: ArrayRef) -> VortexResult<()> {
        let aggregates: Vec<_> = numeric_aggregates()
            .into_iter()
            .filter(|aggregate_fn| aggregate_fn.return_dtype(batch.dtype()).is_some())
            .collect();
        for defined in defined_masks(batch.len()) {
            assert_definedness_matches_mask(&aggregates, &batch, &defined)?;
        }
        Ok(())
    }

    #[test]
    fn definedness_matches_mask_over_floats() -> VortexResult<()> {
        let batch = PrimitiveArray::from_option_iter([
            Some(f64::NAN),
            Some(2.5),
            None,
            Some(f64::NAN),
            Some(-1.0),
            None,
        ])
        .into_array();
        for defined in defined_masks(batch.len()) {
            assert_definedness_matches_mask(&float_aggregates(), &batch, &defined)?;
        }
        Ok(())
    }

    #[test]
    fn definedness_matches_mask_over_strings() -> VortexResult<()> {
        let batch = VarBinViewArray::from_iter_nullable_str([
            Some("zebra"),
            Some("apple"),
            None,
            Some("mango"),
            Some("apple"),
        ])
        .into_array();
        for defined in defined_masks(batch.len()) {
            assert_definedness_matches_mask(&string_aggregates(), &batch, &defined)?;
        }
        Ok(())
    }

    /// Batches with and without partial definedness accumulate into one state, whose result is that
    /// of the batches masked by their definedness masks.
    #[rstest]
    #[case::non_nullable(buffer![5i32, -9, 3, 3, 7, 1].into_array())]
    #[case::nullable(
        PrimitiveArray::from_option_iter([Some(5i64), None, Some(3), Some(3), None, Some(1)])
            .into_array()
    )]
    fn defined_and_undefined_batches_accumulate_together(
        #[case] batch: ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = ctx();
        let len = batch.len();
        let partial = Mask::from_iter((0..len).map(|i| i % 3 != 1));
        let defined_masks = [
            Mask::new_true(len),
            partial.clone(),
            Mask::new_true(len),
            partial,
        ];
        for aggregate_fn in numeric_aggregates()
            .into_iter()
            .filter(|aggregate_fn| aggregate_fn.return_dtype(batch.dtype()).is_some())
        {
            let mut defined_acc = aggregate_fn.accumulator(batch.dtype())?;
            let mut logical = aggregate_fn.accumulator(&batch.dtype().as_nullable())?;
            for defined in &defined_masks {
                defined_acc.accumulate_defined(&batch, defined, &mut ctx)?;
                logical.accumulate(&batch.clone().mask(defined.clone().into_array())?, &mut ctx)?;
            }
            assert_eq!(defined_acc.finish()?, logical.finish()?, "{aggregate_fn}");
        }
        Ok(())
    }

    /// A non-nullable batch of each kind of dtype.
    fn non_nullable_batches(ctx: &mut ExecutionCtx) -> VortexResult<Vec<ArrayRef>> {
        let i32s = buffer![5i32, -9, 3, 3, 7, 1].into_array();
        Ok(vec![
            i32s.clone(),
            buffer![2.5f64, -1.0, 0.0, 9.0, 1.5, 3.0].into_array(),
            BoolArray::from_iter([true, false, true, true, false, false]).into_array(),
            VarBinViewArray::from_iter_str(["a", "bb", "a", "ccc", "d", "bb"]).into_array(),
            DecimalArray::new(
                buffer![100i64, 250, -75, 0, 13, 999],
                DecimalDType::new(10, 2),
                Validity::NonNullable,
            )
            .into_array(),
            StructArray::try_from_iter([("a", i32s.clone())])?.into_array(),
            ListArray::try_new(
                i32s.clone(),
                buffer![0u32, 1, 1, 3, 4, 5, 6].into_array(),
                Validity::NonNullable,
                ctx,
            )?
            .into_array(),
            FixedSizeListArray::try_new(i32s, 1, Validity::NonNullable, 6)?.into_array(),
        ])
    }

    /// An aggregate that keeps the default definedness semantics accumulates a batch masked to the
    /// nullable input dtype into its partial state, so its partial dtype must not depend on the
    /// nullability of its input.
    fn assert_default_definedness_keeps_partial_dtype<V: AggregateFnVTable>(
        vtable: V,
        options: V::Options,
    ) -> VortexResult<()> {
        let mut ctx = ctx();
        for batch in non_nullable_batches(&mut ctx)? {
            let Ok(dtypes) = AggregateDTypes::try_new(&vtable, &options, batch.dtype().clone())
            else {
                continue;
            };
            let args = dtypes.args(&options);
            let mut partial = vtable.empty_partial(args)?;
            let defined = Mask::from_iter((0..batch.len()).map(|i| i % 2 == 0));
            if vtable.try_accumulate_defined(args, &mut partial, &batch, &defined, &mut ctx)? {
                continue;
            }
            assert_eq!(
                vtable.partial_dtype(&options, &batch.dtype().as_nullable()),
                Some(dtypes.partial_dtype.clone()),
                "{} over {}",
                vtable.id(),
                batch.dtype(),
            );
        }
        Ok(())
    }

    #[test]
    fn default_definedness_keeps_partial_dtype_of_every_aggregate() -> VortexResult<()> {
        let opts = NumericalAggregateOpts::default();
        let nans = NumericalAggregateOpts::include_nans();
        let max_bytes = NonZeroUsize::new(2).expect("non zero");
        assert_default_definedness_keeps_partial_dtype(AllNan, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(AllNonDistinct, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(AllNonNan, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(AllNonNull, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(AllNull, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(
            BoundedMax,
            BoundedMaxOptions { max_bytes },
        )?;
        assert_default_definedness_keeps_partial_dtype(
            BoundedMin,
            BoundedMinOptions { max_bytes },
        )?;
        assert_default_definedness_keeps_partial_dtype(Count, opts)?;
        assert_default_definedness_keeps_partial_dtype(Count, nans)?;
        assert_default_definedness_keeps_partial_dtype(First, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(IsConstant, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(
            IsSorted,
            IsSortedOptions { strict: false },
        )?;
        assert_default_definedness_keeps_partial_dtype(IsSorted, IsSortedOptions { strict: true })?;
        assert_default_definedness_keeps_partial_dtype(Last, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(Max, opts)?;
        assert_default_definedness_keeps_partial_dtype(Mean::combined(), PairOptions(opts, opts))?;
        assert_default_definedness_keeps_partial_dtype(Min, opts)?;
        assert_default_definedness_keeps_partial_dtype(MinMax, opts)?;
        assert_default_definedness_keeps_partial_dtype(MinMax, nans)?;
        assert_default_definedness_keeps_partial_dtype(NanCount, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(NullCount, EmptyOptions)?;
        assert_default_definedness_keeps_partial_dtype(Sum, opts)?;
        assert_default_definedness_keeps_partial_dtype(Sum, nans)?;
        assert_default_definedness_keeps_partial_dtype(SumV2, opts)?;
        assert_default_definedness_keeps_partial_dtype(UncompressedSizeInBytes, EmptyOptions)?;
        Ok(())
    }

    /// The uncompressed size measures the stored batch, which definedness does not change.
    #[test]
    fn uncompressed_size_ignores_definedness() -> VortexResult<()> {
        let mut ctx = ctx();
        let size = UncompressedSizeInBytes.bind(EmptyOptions);
        for batch in non_nullable_batches(&mut ctx)? {
            let mut defined_acc = size.accumulator(batch.dtype())?;
            defined_acc.accumulate_defined(
                &batch,
                &Mask::from_iter((0..batch.len()).map(|i| i % 2 == 0)),
                &mut ctx,
            )?;
            let mut stored = size.accumulator(batch.dtype())?;
            stored.accumulate(&batch, &mut ctx)?;
            assert_eq!(defined_acc.finish()?, stored.finish()?, "{}", batch.dtype());
        }
        Ok(())
    }

    #[test]
    fn definedness_length_must_match_batch() -> VortexResult<()> {
        let batch = buffer![1i32, 2, 3].into_array();
        let mut acc = Sum
            .bind(NumericalAggregateOpts::default())
            .accumulator(batch.dtype())?;
        assert!(
            acc.accumulate_defined(&batch, &Mask::new_true(2), &mut ctx())
                .is_err()
        );
        Ok(())
    }

    /// Fully defined batches take the plain path, which caches the result on the batch; partially
    /// defined batches never cache a statistic on the batch.
    #[test]
    fn only_fully_defined_batches_cache_statistics() -> VortexResult<()> {
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;
        use crate::expr::stats::StatsProviderExt;

        let mut ctx = ctx();
        let min = Min.bind(NumericalAggregateOpts::default());

        let partially = buffer![5i32, 1, 3].into_array();
        let mut acc = min.accumulator(partially.dtype())?;
        acc.accumulate_defined(&partially, &Mask::from_iter([true, false, true]), &mut ctx)?;
        assert_eq!(
            acc.finish()?,
            Scalar::primitive(3i32, Nullability::Nullable)
        );
        assert_eq!(
            partially.statistics().get_as::<i32>(Stat::Min),
            Precision::Absent
        );

        let fully = buffer![5i32, 1, 3].into_array();
        acc.accumulate_defined(&fully, &Mask::new_true(3), &mut ctx)?;
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

    /// The defined rows are filtered lazily, and a dictionary filters only its codes, so the
    /// dictionary's kernel still fires.
    #[test]
    fn definedness_keeps_encoding_kernels() -> VortexResult<()> {
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
        acc.accumulate_defined(&batch, &Mask::from_iter([true, false, true]), &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(42i64, Nullability::Nullable)
        );
        Ok(())
    }

    /// Aggregating a struct field under the field definedness matches aggregating the logical field,
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

        let field_definedness =
            struct_field_definedness(&array, &Mask::new_true(array.len()), &mut ctx)?;
        let aggregates = [
            Min.bind(NumericalAggregateOpts::default()),
            Max.bind(NumericalAggregateOpts::default()),
            Sum.bind(NumericalAggregateOpts::default()),
            NullCount.bind(EmptyOptions),
            IsConstant.bind(EmptyOptions),
        ];
        for aggregate_fn in &aggregates {
            let mut defined_acc = aggregate_fn.accumulator(field.dtype())?;
            defined_acc.accumulate_defined(&field, &field_definedness, &mut ctx)?;
            let mut logical = aggregate_fn.accumulator(logical_field.dtype())?;
            logical.accumulate(&logical_field, &mut ctx)?;
            assert_eq!(defined_acc.finish()?, logical.finish()?, "{aggregate_fn}");
        }

        let mut min = aggregates[0].accumulator(field.dtype())?;
        min.accumulate_defined(&field, &field_definedness, &mut ctx)?;
        assert_eq!(
            min.finish()?,
            Scalar::primitive(1i32, Nullability::Nullable)
        );
        // The field's own null, and the two rows the struct nulls out.
        let mut null_count = aggregates[3].accumulator(field.dtype())?;
        null_count.accumulate_defined(&field, &field_definedness, &mut ctx)?;
        assert_eq!(
            null_count.finish()?,
            Scalar::primitive(3u64, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn non_nullable_parents_define_every_row() -> VortexResult<()> {
        let mut ctx = ctx();
        let array = StructArray::try_from_iter([("a", buffer![1i32, 2].into_array())])?;
        assert!(struct_field_definedness(&array, &Mask::new_true(2), &mut ctx)?.all_true());
        Ok(())
    }

    #[test]
    fn struct_field_definedness_intersects_validity() -> VortexResult<()> {
        let array = StructArray::try_from_iter_with_validity(
            [("a", buffer![1i32, 2, 3, 4].into_array())],
            Validity::from_iter([true, false, true, true]),
        )?;
        let defined = Mask::from_iter([true, true, false, true]);

        let field_definedness = struct_field_definedness(&array, &defined, &mut ctx())?;

        assert_eq!(
            field_definedness,
            Mask::from_iter([true, false, false, true])
        );
        Ok(())
    }

    #[test]
    fn list_elements_definedness_skips_null_lists_and_unreferenced_elements() -> VortexResult<()> {
        // Lists [1, 2], [3] (null) and [4, 5] over elements that start with an unreferenced 0.
        let array = ListArray::try_new(
            buffer![0i32, 1, 2, 3, 4, 5, 6].into_array(),
            buffer![1u16, 3, 4, 6].into_array(),
            Validity::from_iter([true, false, true]),
            &mut SESSION.create_execution_ctx(),
        )?;

        let elements_definedness =
            list_elements_definedness(&array, &Mask::new_true(3), &mut ctx())?;

        assert_eq!(
            elements_definedness,
            Mask::from_iter([false, true, true, false, true, true, false])
        );
        Ok(())
    }

    /// The elements of the defined, valid lists, computed one element at a time.
    fn reference_elements_definedness(
        ranges: &[(u32, u32)],
        lists: &Mask,
        elements_end: u32,
    ) -> Mask {
        Mask::from_iter((0..elements_end).map(|element| {
            ranges
                .iter()
                .enumerate()
                .any(|(row, &(start, end))| lists.value(row) && start <= element && element < end)
        }))
    }

    /// Run-based list element definedness agrees with the element-by-element reference, across empty
    /// and null lists and partial definedness masks.
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    fn elements_definedness_matches_reference(#[case] seed: u64) -> VortexResult<()> {
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
        let defined = Mask::from_iter((0..len).map(|_| rng.random_bool(0.8)));
        let expected = reference_elements_definedness(&ranges, &(&valid & &defined), elements_end);
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
        assert_eq!(
            list_elements_definedness(&list, &defined, &mut ctx)?,
            expected
        );
        Ok(())
    }
}
