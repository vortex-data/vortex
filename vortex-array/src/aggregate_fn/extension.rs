// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic adapter that defines an aggregate over one extension type by running an existing
//! aggregate over the extension's storage.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateDTypes;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::DynGroupedAccumulator;
use crate::aggregate_fn::GroupedAccumulator;
use crate::aggregate_fn::GroupedArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::Scalar;

/// The aggregate `A` applied to one extension type through its storage.
///
/// Inputs must have an extension dtype of that type. Every batch is unwrapped to its storage array
/// and accumulated by `A`, so storage-encoding kernels registered for `A` still fire, grouped ones
/// included.
///
/// When `A` defines its partial and return dtypes over the extension dtype itself, as min/max does
/// (`{min, max}` of the input dtype), those are this aggregate's dtypes, and the storage partials
/// and results of `A` are read as extension values. Otherwise they are `A`'s dtypes over the
/// storage dtype, e.g. the integer sum of timestamps.
///
/// The adapter is an aggregate of its own, with id `<A>[<E>]`, so supporting a new pair is one
/// registration:
///
/// ```
/// # use vortex_array::aggregate_fn::ExtensionAggregate;
/// # use vortex_array::aggregate_fn::fns::sum::Sum;
/// # use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
/// # use vortex_array::extension::datetime::Timestamp;
/// # let session = vortex_array::array_session();
/// session.aggregate_fns().register(ExtensionAggregate::new(Sum, Timestamp));
/// ```
///
/// Only register a pair when `A` over the storage is meaningful for the extension type, e.g. when
/// it orders and compares like its storage.
///
/// Only the aggregate is generic: the adapter implements [`AggregateFnVTable`] with `A`'s options
/// and partials, while the extension type is identified by its [`ExtId`] alone, so aggregates can
/// build the adapter for whichever extension dtype they are given.
#[derive(Clone, Debug)]
pub struct ExtensionAggregate<A: AggregateFnVTable> {
    aggregate: A,
    ext_id: ExtId,
    id: AggregateFnId,
}

impl<A: AggregateFnVTable> ExtensionAggregate<A> {
    /// Define `aggregate` over extension type `ext`.
    pub fn new(aggregate: A, ext: impl ExtVTable) -> Self {
        Self::for_ext_id(aggregate, ext.id())
    }

    /// Define `aggregate` over the extension type with id `ext_id`.
    pub fn for_ext_id(aggregate: A, ext_id: ExtId) -> Self {
        #[expect(
            clippy::disallowed_methods,
            reason = "the id is built from the aggregate and extension ids"
        )]
        let id = AggregateFnId::new(&format!("{}[{ext_id}]", aggregate.id()));
        Self {
            aggregate,
            ext_id,
            id,
        }
    }

    /// The storage dtype of `dtype`, if it is an extension dtype of this type.
    fn storage_dtype<'a>(&self, dtype: &'a DType) -> Option<&'a DType> {
        dtype
            .as_extension_opt()
            .filter(|ext_dtype| ext_dtype.id() == self.ext_id)
            .map(|ext_dtype| ext_dtype.storage_dtype())
    }

    /// The dtype `A` is resolved over for `input_dtype`: the extension dtype itself if `A` defines
    /// its dtypes over it, otherwise its storage dtype.
    fn resolved_dtype<'a>(
        &self,
        options: &A::Options,
        input_dtype: &'a DType,
    ) -> Option<&'a DType> {
        let storage_dtype = self.storage_dtype(input_dtype)?;
        let over_ext = self.aggregate.return_dtype(options, input_dtype).is_some()
            && self.aggregate.partial_dtype(options, input_dtype).is_some();
        Some(if over_ext { input_dtype } else { storage_dtype })
    }

    /// The dtypes of the storage aggregate `A` over the storage dtype.
    fn storage_dtypes(&self, args: AggregateArgs<'_, A::Options>) -> VortexResult<AggregateDTypes> {
        let storage_dtype = self.storage_dtype(args.dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} expects extension type {}, got {}",
                self.id,
                self.ext_id,
                args.dtype
            )
        })?;
        AggregateDTypes::try_new(&self.aggregate, args.options, storage_dtype.clone())
    }
}

/// The storage of an extension-typed columnar array.
fn columnar_storage(batch: &Columnar) -> VortexResult<ArrayRef> {
    match batch {
        Columnar::Canonical(Canonical::Extension(array)) => Ok(array.storage_array().clone()),
        Columnar::Constant(constant) => Ok(ConstantArray::new(
            constant.scalar().as_extension().to_storage_scalar(),
            constant.len(),
        )
        .into_array()),
        Columnar::Canonical(_) => {
            vortex_bail!("Expected an extension array, got {}", batch.dtype())
        }
    }
}

/// Read `scalar` as a scalar of `dtype`, which differs from its dtype only by storage dtypes read
/// as their extension dtypes or the reverse.
///
/// Extension scalars hold storage values, so the value is unchanged.
fn retype_scalar(scalar: Scalar, dtype: &DType) -> VortexResult<Scalar> {
    if scalar.dtype() == dtype {
        return Ok(scalar);
    }
    Scalar::try_new(dtype.clone(), scalar.into_value())
}

/// Read `array` as an array of `dtype`, which differs from its dtype only by storage dtypes read as
/// their extension dtypes or the reverse.
///
/// Aggregate states are built as extension, struct and chunked arrays, so this rewraps those
/// structurally without executing anything.
fn retype_array(array: ArrayRef, dtype: &DType) -> VortexResult<ArrayRef> {
    let array_dtype = array.dtype().clone();
    if &array_dtype == dtype {
        return Ok(array);
    }
    if let DType::Extension(ext_dtype) = dtype
        && ext_dtype.storage_dtype() == &array_dtype
    {
        return Ok(ExtensionArray::try_new(ext_dtype.clone(), array)?.into_array());
    }
    if let Some(array) = array.as_opt::<Extension>()
        && array.storage_array().dtype() == dtype
    {
        return Ok(array.storage_array().clone());
    }
    if let Some(chunked) = array.as_opt::<Chunked>() {
        let chunks = chunked
            .iter_chunks()
            .map(|chunk| retype_array(chunk.clone(), dtype))
            .collect::<VortexResult<Vec<_>>>()?;
        return Ok(ChunkedArray::try_new(chunks, dtype.clone())?.into_array());
    }
    if let DType::Struct(fields, _) = dtype
        && let Some(array) = array.as_opt::<Struct>()
        && array.struct_fields().nfields() == fields.nfields()
    {
        let retyped = array
            .iter_unmasked_fields()
            .zip(fields.fields())
            .map(|(field, dtype)| retype_array(field.clone(), &dtype))
            .collect::<VortexResult<Vec<_>>>()?;
        return Ok(StructArray::try_new_with_dtype(
            retyped,
            fields.clone(),
            array.len(),
            array.struct_validity(),
        )?
        .into_array());
    }
    vortex_bail!("Cannot read {array_dtype} aggregate states as {dtype}")
}

impl<A: AggregateFnVTable> AggregateFnVTable for ExtensionAggregate<A> {
    type Options = A::Options;
    // The storage aggregate is held as a fully-fledged `Accumulator` so that storage batches
    // consult the kernel registry and the cached statistics of `A`.
    type Partial = Accumulator<A>;

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
        self.aggregate
            .return_dtype(options, self.resolved_dtype(options, input_dtype)?)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.aggregate
            .partial_dtype(options, self.resolved_dtype(options, input_dtype)?)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        Ok(Accumulator::from_dtypes(
            self.aggregate.clone(),
            args.options.clone(),
            self.storage_dtypes(args)?,
        ))
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        let storage_partial_dtype = self.storage_dtypes(args)?.partial_dtype;
        let mut partial = self.empty_partial(args)?;
        partial.combine_partials(&retype_scalar(scalar.clone(), &storage_partial_dtype)?)?;
        Ok(partial)
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        mut second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        first.merge_from(&mut second)?;
        Ok(first)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        retype_scalar(partial.partial_scalar()?, args.partial_dtype)
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        partial.is_saturated()
    }

    /// Unwrap extension arrays to their storage before decoding, so the storage accumulator sees
    /// the storage encoding.
    fn try_accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(array) = batch.as_opt::<Extension>() else {
            return Ok(false);
        };
        state.accumulate(array.storage_array(), ctx)?;
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        state.accumulate(&columnar_storage(batch)?, ctx)
    }

    /// Aggregate the groups over their storage elements with a grouped accumulator of `A`, so
    /// grouped kernels for the storage encoding fire.
    fn try_accumulate_grouped(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let elements = groups.elements().clone().execute::<Columnar>(ctx)?;
        // SAFETY: we preserve the grouped shape and validity while replacing the elements with
        // their storage, so the groups become groups of the storage dtype.
        let storage_groups =
            unsafe { groups.with_elements_unchecked(columnar_storage(&elements)?)? };
        let mut storage_acc = GroupedAccumulator::from_dtypes(
            self.aggregate.clone(),
            args.options.clone(),
            self.storage_dtypes(args)?,
        );
        storage_acc.accumulate_grouped_array(storage_groups, ctx)?;
        Ok(Some(retype_array(
            storage_acc.flush()?,
            args.partial_dtype,
        )?))
    }

    fn finalize(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        states: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let dtypes = self.storage_dtypes(args)?;
        let states = retype_array(states, &dtypes.partial_dtype)?;
        let results = self.aggregate.finalize(dtypes.args(args.options), states)?;
        retype_array(results, args.return_dtype)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        retype_scalar(partial.final_scalar()?, args.return_dtype)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use parking_lot::Mutex;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::ExtensionAggregate;
    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::GroupedArray;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::min_max::MinMaxResult;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::kernels::DynGroupedAggregateKernel;
    use crate::aggregate_fn::session::AggregateFnSessionExt;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::extension::tests::divisible_int::DivisibleInt;
    use crate::extension::tests::divisible_int::Divisor;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn sums_extension_through_storage() -> VortexResult<()> {
        let session = crate::array_session();
        let aggregate = ExtensionAggregate::new(Sum, Timestamp);
        session.aggregate_fns().register(aggregate.clone());
        assert!(
            session
                .aggregate_fns()
                .find_plugin(&aggregate.id())
                .is_some()
        );

        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let storage =
            PrimitiveArray::from_option_iter([Some(3_000i64), None, Some(1_000)]).into_array();
        let array = ExtensionArray::new(ext_dtype.clone(), storage.clone()).into_array();

        let mut ctx = session.create_execution_ctx();
        let mut acc = Accumulator::try_new(
            aggregate,
            NumericalAggregateOpts::default(),
            DType::Extension(ext_dtype),
        )?;
        acc.accumulate(&array, &mut ctx)?;

        let mut expected = Accumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            storage.dtype().clone(),
        )?;
        expected.accumulate(&storage, &mut ctx)?;

        assert_eq!(acc.finish()?, expected.finish()?);
        Ok(())
    }

    /// Registering `ExtensionAggregate<MinMax>` gives an extension type extrema, as extension
    /// scalars.
    #[test]
    fn registered_min_max_gives_extension_extrema() -> VortexResult<()> {
        let session = crate::array_session();
        let mut ctx = session.create_execution_ctx();
        let opts = NumericalAggregateOpts::default();

        let ext_dtype = ExtDType::<DivisibleInt>::try_new(
            Divisor(5),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
        let ext_scalar = |value: u64| Scalar::extension_ref(ext_dtype.clone(), Scalar::from(value));
        let array =
            ExtensionArray::new(ext_dtype.clone(), buffer![15u64, 5, 10].into_array()).into_array();
        let constant = ConstantArray::new(ext_scalar(20), 2).into_array();

        assert_eq!(min_max(&array, &mut ctx, opts)?, None);
        assert_eq!(min_max(&constant, &mut ctx, opts)?, None);

        session
            .aggregate_fns()
            .register(ExtensionAggregate::new(MinMax, DivisibleInt));
        assert_eq!(
            min_max(&array, &mut ctx, opts)?,
            Some(MinMaxResult {
                min: ext_scalar(5),
                max: ext_scalar(15),
            })
        );
        assert_eq!(
            min_max(&constant, &mut ctx, opts)?,
            Some(MinMaxResult {
                min: ext_scalar(20),
                max: ext_scalar(20),
            })
        );
        Ok(())
    }

    #[test]
    fn rejects_other_extension_types() {
        let date = DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased());
        assert!(
            ExtensionAggregate::new(Sum, Timestamp)
                .return_dtype(&NumericalAggregateOpts::default(), &date)
                .is_none()
        );
    }

    /// Aggregates that define their dtypes over the extension dtype keep them; others take their
    /// dtypes over the storage dtype.
    #[test]
    fn dtypes_follow_the_aggregate() {
        let opts = NumericalAggregateOpts::default();
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let input = DType::Extension(ext_dtype.clone());
        let storage = ext_dtype.storage_dtype();

        let min_max = ExtensionAggregate::new(MinMax, Timestamp);
        assert_eq!(
            min_max.return_dtype(&opts, &input),
            MinMax.return_dtype(&opts, &input)
        );
        assert_eq!(
            min_max.partial_dtype(&opts, &input),
            MinMax.partial_dtype(&opts, &input)
        );

        let sum = ExtensionAggregate::new(Sum, Timestamp);
        assert_eq!(
            sum.return_dtype(&opts, &input),
            Sum.return_dtype(&opts, storage)
        );
        assert_eq!(
            sum.partial_dtype(&opts, &input),
            Sum.partial_dtype(&opts, storage)
        );
    }

    /// Records the element dtypes a grouped min/max kernel is offered, then declines.
    #[derive(Debug, Default)]
    struct ProbeGroupedKernel(Mutex<Vec<DType>>);

    impl DynGroupedAggregateKernel for ProbeGroupedKernel {
        fn grouped_aggregate(
            &self,
            _aggregate_fn: &AggregateFnRef,
            groups: &GroupedArray,
            _ctx: &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>> {
            self.0.lock().push(groups.elements().dtype().clone());
            Ok(None)
        }
    }

    /// Grouped min/max over a registered extension type runs over the grouped storage, offering
    /// it to grouped kernels, and gives extension-typed extrema.
    #[test]
    fn grouped_min_max_runs_over_storage() -> VortexResult<()> {
        static PROBE: LazyLock<ProbeGroupedKernel> = LazyLock::new(ProbeGroupedKernel::default);

        let session = crate::array_session();
        session
            .aggregate_fns()
            .register_grouped_kernel(MinMax.id(), &*PROBE);
        session
            .aggregate_fns()
            .register(ExtensionAggregate::new(MinMax, DivisibleInt));
        let mut ctx = session.create_execution_ctx();
        let opts = NumericalAggregateOpts::default();

        let ext_dtype = ExtDType::<DivisibleInt>::try_new(
            Divisor(5),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
        let input = DType::Extension(ext_dtype.clone());
        let ext_scalar = |value: u64| Scalar::extension_ref(ext_dtype.clone(), Scalar::from(value));
        let elements = ExtensionArray::new(
            ext_dtype.clone(),
            buffer![15u64, 5, 10, 20, 30, 25].into_array(),
        )
        .into_array();
        let groups =
            FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 3)?.into_array();
        let expected = [(5, 15), (10, 20), (25, 30)].map(|(min, max)| {
            Some(MinMaxResult {
                min: ext_scalar(min),
                max: ext_scalar(max),
            })
        });

        let aggregates = [
            MinMax.bind(opts),
            ExtensionAggregate::new(MinMax, DivisibleInt).bind(opts),
        ];
        for aggregate in aggregates {
            PROBE.0.lock().clear();
            let mut acc = aggregate.accumulator_grouped(&input)?;
            acc.accumulate_list(&groups, &mut ctx)?;
            let results = acc.finish()?;
            let results = (0..results.len())
                .map(|row| MinMaxResult::from_scalar(&results.execute_scalar(row, &mut ctx)?))
                .collect::<VortexResult<Vec<_>>>()?;
            assert_eq!(results, expected, "{aggregate}");
            assert!(
                PROBE.0.lock().contains(ext_dtype.storage_dtype()),
                "{aggregate} offers grouped kernels the storage elements"
            );
        }
        Ok(())
    }
}
