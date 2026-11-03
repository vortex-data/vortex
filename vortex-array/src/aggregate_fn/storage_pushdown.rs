// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opt-in pushdown of an aggregate into the storage of one extension type.
//!
//! See [`ExtensionStoragePushdown`] for when such a pushdown is correct.

use std::mem;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::dtype::StructFields;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;
use crate::scalar::Scalar;

/// Runs aggregate `A` on the storage values of extension type `E`.
///
/// The input must have extension type `E`, and `A` must support its storage dtype. Binding fails
/// for any other input. The wrapper has a separate aggregate ID, `<A>[<E>]`, formed from the IDs
/// of `A` and `E`. Registering it does not replace `A`.
///
/// Aggregates that already support extension dtypes directly do not need this wrapper.
///
/// # Correctness
///
/// A registration **must** satisfy both conditions:
///
/// - The aggregation is defined for logical values of `E`.
/// - For every valid input, aggregating the storage values produces the storage representation
///   of the logical aggregate result. This requires storage to preserve whichever properties `A`
///   relies on, such as ordering, equality, arithmetic, or NaN classification.
///
/// For example, minimum is defined for timestamps, and their integer storage preserves their
/// ordering, so both conditions hold. Summing timestamps fails the first condition even though
/// their storage supports addition.
///
/// These conditions are not checked. An invalid registration can silently return incorrect results.
///
/// # Results and partials
///
/// Results and partial states are those of `A` over the storage dtype, except that their value
/// fields, named by [`StorageResultMapping`], hold values of `E`. For example, the minimum and
/// maximum of a timestamp column are timestamps, as with plain `min_max`.
///
/// Partial states can only be combined when the aggregate, options, and full input dtype match.
/// This includes extension metadata, so timestamp partials with different time units must not be
/// combined even if their storage dtypes match.
///
/// # Example
///
/// Timestamps compare like their `i64` storage, so `is_sorted` can be pushed down. Register the
/// pair once on the session, then bind it like any other aggregate:
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_session::VortexSession;
/// # use vortex_array::VortexSessionExecute;
/// # use vortex_array::aggregate_fn::Accumulator;
/// # use vortex_array::aggregate_fn::DynAccumulator;
/// # use vortex_array::aggregate_fn::ExtensionStoragePushdown;
/// # use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
/// # use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
/// # use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
/// # use vortex_array::arrays::ExtensionArray;
/// # use vortex_array::arrays::PrimitiveArray;
/// # use vortex_array::dtype::DType;
/// # use vortex_array::dtype::Nullability;
/// # use vortex_array::extension::datetime::TimeUnit;
/// # use vortex_array::extension::datetime::Timestamp;
/// # fn main() -> vortex_error::VortexResult<()> {
/// let session = VortexSession::empty();
///
/// // Register `vortex.is_sorted[vortex.timestamp]`.
/// let is_sorted = ExtensionStoragePushdown::new(IsSorted, Timestamp);
/// session.aggregate_fns().register(is_sorted.clone());
///
/// // Aggregate a timestamp column through its storage.
/// let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
/// let column = ExtensionArray::new(
///     ext_dtype.clone(),
///     PrimitiveArray::from_iter([1_000i64, 2_000, 3_000]).into_array(),
/// )
/// .into_array();
///
/// let mut acc = Accumulator::try_new(
///     is_sorted,
///     IsSortedOptions { strict: true },
///     DType::Extension(ext_dtype),
/// )?;
/// acc.accumulate(&column, &mut session.create_execution_ctx())?;
/// assert!(bool::try_from(&acc.finish()?)?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct ExtensionStoragePushdown<A: StorageResultMapping, E: ExtVTable> {
    aggregate: A,
    ext: E,
    id: AggregateFnId,
}

/// An aggregate that names the _value fields_ of its partial and result, which hold values of its
/// input.
///
/// [`ExtensionStoragePushdown`] runs the aggregate on storage values, so the value fields hold
/// storage values, and the pushdown gives them the extension dtype back. A value field is named by
/// its index in a struct dtype because its dtype cannot identify it: over `Bool` storage, the
/// `is_sorted` field of [`IsSorted`] has the storage dtype but holds no input value. The indices
/// **must** name exactly the fields that hold input values. An index out of range, or any index
/// into a dtype that is not a struct, panics when the pushdown resolves its dtypes, and an index on
/// a field of another dtype can fail when the pushdown converts a partial to a scalar. Other
/// mistakes, such as a missing index, silently give a field the wrong dtype.
///
/// The pushdown calls [`finalize`] with its own partial states and dtypes: the value fields hold
/// extension values, and `args.dtype` is the extension dtype. So `finalize` **must** only move or
/// drop the value fields, and it **must not** depend on `args.dtype`.
///
/// [`finalize`]: AggregateFnVTable::finalize
/// [`IsSorted`]: crate::aggregate_fn::fns::is_sorted::IsSorted
pub trait StorageResultMapping: AggregateFnVTable {
    /// The indices of the value fields of the partial dtype.
    const PARTIAL_VALUE_FIELDS: &'static [usize];
    /// The indices of the value fields of the return dtype.
    const RETURN_VALUE_FIELDS: &'static [usize];
}

impl<A: StorageResultMapping, E: ExtVTable> ExtensionStoragePushdown<A, E> {
    /// Push `aggregate` down into the storage of extension type `ext`.
    ///
    /// Only do this when `aggregate` over the storage is correct for `ext`; see
    /// [`ExtensionStoragePushdown`] for how to check.
    pub fn new(aggregate: A, ext: E) -> Self {
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let id = AggregateFnId::new(&format!("{}[{}]", aggregate.id(), ext.id()));
        Self { aggregate, ext, id }
    }

    /// The extension dtype of `dtype`, if it is an extension dtype of type `E`.
    fn ext_dtype<'a>(&self, dtype: &'a DType) -> Option<&'a ExtDTypeRef> {
        dtype
            .as_extension_opt()
            .filter(|ext_dtype| ext_dtype.is::<E>())
    }

    /// Resolve the dtypes of the storage aggregate `A` over the storage dtype of `args.dtype`.
    ///
    /// The value fields of the return and partial dtypes in `args` have the extension dtype, so `A`
    /// needs dtypes of its own.
    fn storage_dtypes(&self, args: AggregateArgs<'_, A::Options>) -> VortexResult<AggregateDTypes> {
        let ext_dtype = self.ext_dtype(args.dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} expects extension type {}, got {}",
                self.id,
                self.ext.id(),
                args.dtype
            )
        })?;
        AggregateDTypes::try_new(
            &self.aggregate,
            args.options,
            ext_dtype.storage_dtype().clone(),
        )
    }

    /// Accumulate a storage batch into `state` through an [`Accumulator`] of `A`, so that the
    /// kernels and cached statistics of `A` on the storage encoding still apply.
    fn accumulate_storage(
        &self,
        args: AggregateArgs<'_, A::Options>,
        state: &mut A::Partial,
        storage: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut accumulator = Accumulator::from_dtypes(
            self.aggregate.clone(),
            args.options.clone(),
            AggregateDTypes::new(
                args.dtype.clone(),
                args.return_dtype.clone(),
                args.partial_dtype.clone(),
            ),
        );
        accumulator.accumulate(storage, ctx)?;
        let Some(batch) = accumulator.take_partial() else {
            return Ok(());
        };
        let running = mem::replace(state, self.aggregate.empty_partial(args)?);
        *state = self.aggregate.merge_partials(args, running, batch)?;
        Ok(())
    }
}

impl<A: StorageResultMapping, E: ExtVTable> AggregateFnVTable for ExtensionStoragePushdown<A, E> {
    type Options = A::Options;
    type Partial = A::Partial;

    fn id(&self) -> AggregateFnId {
        self.id
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        self.aggregate.serialize(options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        self.aggregate.deserialize(metadata, session)
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let ext_dtype = self.ext_dtype(input_dtype)?;
        self.aggregate
            .return_dtype(options, ext_dtype.storage_dtype())
            .map(|dtype| wrap_value_fields(dtype, A::RETURN_VALUE_FIELDS, ext_dtype))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let ext_dtype = self.ext_dtype(input_dtype)?;
        self.aggregate
            .partial_dtype(options, ext_dtype.storage_dtype())
            .map(|dtype| wrap_value_fields(dtype, A::PARTIAL_VALUE_FIELDS, ext_dtype))
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        let storage_dtypes = self.storage_dtypes(args)?;
        self.aggregate
            .empty_partial(storage_dtypes.args(args.options))
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        let storage_dtypes = self.storage_dtypes(args)?;

        // An extension value is stored as its storage value, so only the dtype of the partial
        // changes. A null partial and a null field stay null.
        let storage_scalar = Scalar::try_new(
            storage_dtypes.partial_dtype.clone(),
            scalar.value().cloned(),
        )?;

        self.aggregate
            .partial_from_scalar(storage_dtypes.args(args.options), &storage_scalar)
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        let storage_dtypes = self.storage_dtypes(args)?;
        self.aggregate
            .merge_partials(storage_dtypes.args(args.options), first, second)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let storage_dtypes = self.storage_dtypes(args)?;
        let storage_scalar = self
            .aggregate
            .to_scalar(storage_dtypes.args(args.options), partial)?;

        // The reverse of `partial_from_scalar`. `Scalar::try_new` also runs the value check of `E`,
        // as plain `min_max` does when it wraps its extrema.
        Scalar::try_new(args.partial_dtype.clone(), storage_scalar.into_value())
    }

    fn is_saturated(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        self.storage_dtypes(args).is_ok_and(|storage_dtypes| {
            self.aggregate
                .is_saturated(storage_dtypes.args(args.options), partial)
        })
    }

    /// Unwrap extension arrays to their storage before decoding, so the storage accumulator sees
    /// the storage encoding.
    fn try_accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(array) = batch.as_opt::<Extension>() else {
            return Ok(false);
        };
        let storage_dtypes = self.storage_dtypes(args)?;
        self.accumulate_storage(
            storage_dtypes.args(args.options),
            state,
            array.storage_array(),
            ctx,
        )?;
        Ok(true)
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let storage_dtypes = self.storage_dtypes(args)?;
        let args = storage_dtypes.args(args.options);
        match batch {
            Columnar::Canonical(Canonical::Extension(array)) => {
                self.accumulate_storage(args, state, array.storage_array(), ctx)
            }
            // A constant storage array has no encoding to dispatch on and no statistics worth
            // caching, so it goes straight to `A` rather than through a nested accumulator.
            Columnar::Constant(constant) => {
                let storage = constant.scalar().as_extension().to_storage_scalar();
                self.aggregate.accumulate(
                    args,
                    state,
                    &Columnar::Constant(ConstantArray::new(storage, constant.len())),
                    ctx,
                )
            }
            Columnar::Canonical(_) => {
                vortex_bail!(
                    "Aggregate function {} expects an extension array, got {}",
                    self.id,
                    batch.dtype()
                )
            }
        }
    }

    fn finalize(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        // The states have the partial dtype of this aggregate, so their value fields hold values
        // of `E`. `StorageResultMapping` requires that `A` finalizes them as they are, with these
        // `args`.
        self.aggregate.finalize(args, states)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let storage_dtypes = self.storage_dtypes(args)?;
        let storage_result = self
            .aggregate
            .finalize_scalar(storage_dtypes.args(args.options), partial)?;

        Scalar::try_new(args.return_dtype.clone(), storage_result.into_value())
    }
}

/// Give the fields `value_fields` of the struct `dtype` the extension dtype `ext_dtype`, with the
/// nullability of each field.
fn wrap_value_fields(dtype: DType, value_fields: &[usize], ext_dtype: &ExtDTypeRef) -> DType {
    if value_fields.is_empty() {
        return dtype;
    }

    let struct_fields = dtype.as_struct_fields();
    let mut field_dtypes: Vec<DType> = struct_fields.fields().collect();
    for &idx in value_fields {
        let nullability = field_dtypes[idx].nullability();
        field_dtypes[idx] = DType::Extension(ext_dtype.with_nullability(nullability));
    }

    DType::Struct(
        StructFields::new(struct_fields.names().clone(), field_dtypes),
        dtype.nullability(),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ExtensionStoragePushdown;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::is_sorted::make_is_sorted_partial_dtype;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::session::AggregateFnSessionExt;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// A column of nullable timestamps in milliseconds.
    fn timestamps<const N: usize>(values: [Option<i64>; N]) -> ArrayRef {
        ExtensionArray::new(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased(),
            PrimitiveArray::from_option_iter(values).into_array(),
        )
        .into_array()
    }

    /// Accumulate `batches` with `min_max` pushed down into the storage of `ext` and with plain
    /// `min_max`, and check that both give the same partial and the same result.
    #[track_caller]
    fn assert_matches_plain_min_max(ext: impl ExtVTable, batches: &[ArrayRef]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = batches[0].dtype();
        let mut pushdown = Accumulator::try_new(
            ExtensionStoragePushdown::new(MinMax, ext),
            NumericalAggregateOpts::default(),
            dtype.clone(),
        )?;
        let mut plain =
            Accumulator::try_new(MinMax, NumericalAggregateOpts::default(), dtype.clone())?;
        for batch in batches {
            pushdown.accumulate(batch, &mut ctx)?;
            plain.accumulate(batch, &mut ctx)?;
        }

        assert_eq!(pushdown.partial_scalar()?, plain.partial_scalar()?);
        assert_eq!(pushdown.finish()?, plain.finish()?);
        Ok(())
    }

    /// A timestamp column whose `len` rows all hold `value`.
    fn constant_timestamps(value: i64, len: usize) -> ArrayRef {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let scalar =
            Scalar::extension_ref(ext_dtype, Scalar::primitive(value, Nullability::Nullable));

        ConstantArray::new(scalar, len).into_array()
    }

    /// A timestamp column whose storage is dictionary encoded, so `min_max` runs the kernel of the
    /// dictionary over the storage dtypes.
    fn dictionary_timestamps() -> ArrayRef {
        let storage = DictArray::new(
            buffer![1u32, 0, 2, 1].into_array(),
            PrimitiveArray::from_option_iter([Some(3_000i64), None, Some(1_000)]).into_array(),
        );

        ExtensionArray::new(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased(),
            storage.into_array(),
        )
        .into_array()
    }

    #[rstest]
    #[case::flat(vec![timestamps([Some(3_000), None, Some(1_000)])])]
    // The `Chunked` kernel sends the partial of the chunked batch out through `to_scalar` and back
    // through `partial_from_scalar`, before the next batch merges into it.
    #[case::chunked(vec![
        ChunkedArray::from_iter([timestamps([Some(3_000), None]), timestamps([Some(1_000)])])
            .into_array(),
        timestamps([Some(500)]),
    ])]
    #[case::constant(vec![constant_timestamps(2_000, 3), constant_timestamps(1_000, 3)])]
    #[case::dictionary(vec![dictionary_timestamps()])]
    fn timestamp_min_max_matches_plain_min_max(#[case] batches: Vec<ArrayRef>) -> VortexResult<()> {
        assert_matches_plain_min_max(Timestamp, &batches)
    }

    #[test]
    fn combines_flushed_partials() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let batch = |values: [Option<i64>; 2]| {
            ExtensionArray::new(
                ext_dtype.clone(),
                PrimitiveArray::from_option_iter(values).into_array(),
            )
            .into_array()
        };
        let accumulator = || {
            Accumulator::try_new(
                ExtensionStoragePushdown::new(MinMax, Timestamp),
                NumericalAggregateOpts::default(),
                DType::Extension(ext_dtype.clone()),
            )
        };

        let mut first = accumulator()?;
        first.accumulate(&batch([Some(3_000), None]), &mut ctx)?;
        let mut second = accumulator()?;
        second.accumulate(&batch([Some(1_000), Some(500)]), &mut ctx)?;
        first.combine_partials(&second.flush()?)?;

        let mut expected = accumulator()?;
        expected.accumulate(&batch([Some(3_000), None]), &mut ctx)?;
        expected.accumulate(&batch([Some(1_000), Some(500)]), &mut ctx)?;

        assert_eq!(first.finish()?, expected.finish()?);
        Ok(())
    }

    #[test]
    fn rejects_other_extension_types() {
        let date = DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased());
        assert!(
            ExtensionStoragePushdown::new(MinMax, Timestamp)
                .return_dtype(&NumericalAggregateOpts::default(), &date)
                .is_none()
        );
    }

    #[test]
    fn default_session_registers_temporal_order_aggregates() {
        let session = VortexSession::empty();
        let aggregate_fns = session.aggregate_fns();
        for id in [
            ExtensionStoragePushdown::new(IsSorted, Date).id(),
            ExtensionStoragePushdown::new(IsSorted, Time).id(),
            ExtensionStoragePushdown::new(IsSorted, Timestamp).id(),
            ExtensionStoragePushdown::new(MinMax, Date).id(),
            ExtensionStoragePushdown::new(MinMax, Time).id(),
            ExtensionStoragePushdown::new(MinMax, Timestamp).id(),
        ] {
            assert!(
                aggregate_fns.find_plugin(&id).is_some(),
                "{id} is not registered"
            );
        }
    }

    #[rstest]
    // Nulls sort first.
    #[case::sorted([None, Some(1_000), Some(2_000), Some(2_000)], false, true)]
    #[case::unsorted([Some(2_000), Some(1_000), None, Some(3_000)], false, false)]
    #[case::strict_with_duplicates([None, Some(1_000), Some(2_000), Some(2_000)], true, false)]
    fn is_sorted_over_timestamps(
        #[case] values: [Option<i64>; 4],
        #[case] strict: bool,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let array = ExtensionArray::new(
            ext_dtype.clone(),
            PrimitiveArray::from_option_iter(values).into_array(),
        )
        .into_array();

        let mut acc = Accumulator::try_new(
            ExtensionStoragePushdown::new(IsSorted, Timestamp),
            IsSortedOptions { strict },
            DType::Extension(ext_dtype),
        )?;
        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(bool::try_from(&acc.finish()?)?, expected);
        Ok(())
    }

    /// The `Chunked` kernel sends the partial of the chunked batch out through `to_scalar` and back
    /// through `partial_from_scalar`. Its first value is null, and its last value must come back
    /// as a storage value that the second batch can compare with.
    #[test]
    fn is_sorted_over_chunked_timestamps() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunked =
            ChunkedArray::from_iter([timestamps([None, Some(1_000)]), timestamps([Some(2_000)])])
                .into_array();

        let mut acc = Accumulator::try_new(
            ExtensionStoragePushdown::new(IsSorted, Timestamp),
            IsSortedOptions { strict: false },
            chunked.dtype().clone(),
        )?;
        acc.accumulate(&chunked, &mut ctx)?;
        acc.accumulate(&timestamps([Some(3_000)]), &mut ctx)?;

        assert_eq!(
            acc.partial_scalar()?.dtype(),
            &make_is_sorted_partial_dtype(chunked.dtype())
        );
        assert!(bool::try_from(&acc.finish()?)?);
        Ok(())
    }

    #[test]
    fn min_max_over_dates_matches_plain_min_max() -> VortexResult<()> {
        let dates = ExtensionArray::new(
            Date::new(TimeUnit::Days, Nullability::Nullable).erased(),
            PrimitiveArray::from_option_iter([Some(20i32), None, Some(-3), Some(7)]).into_array(),
        );
        assert_matches_plain_min_max(Date, &[dates.into_array()])
    }

    /// Grouped accumulation finalizes partial states that already hold timestamps.
    #[test]
    fn grouped_min_max_matches_plain_min_max() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = timestamps([Some(3_000), None, Some(1_000), Some(2_000)]);
        let groups =
            FixedSizeListArray::new(elements.clone(), 2, Validity::NonNullable, 2).into_array();

        let mut pushdown = GroupedAccumulator::try_new(
            ExtensionStoragePushdown::new(MinMax, Timestamp),
            NumericalAggregateOpts::default(),
            elements.dtype().clone(),
        )?;
        pushdown.accumulate_list(&groups, &mut ctx)?;
        let mut plain = GroupedAccumulator::try_new(
            MinMax,
            NumericalAggregateOpts::default(),
            elements.dtype().clone(),
        )?;
        plain.accumulate_list(&groups, &mut ctx)?;

        assert_arrays_eq!(pushdown.finish()?, plain.finish()?, &mut ctx);
        Ok(())
    }
}
