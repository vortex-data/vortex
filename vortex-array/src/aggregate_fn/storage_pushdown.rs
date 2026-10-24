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
/// Results and partial states are those of `A` over the storage dtype. Results are not converted
/// back to `E`. For example, the minimum and maximum of a timestamp column are raw `i64` values
/// in the column's time unit.
///
/// Partial states can only be combined when the aggregate, options, and full input dtype match.
/// This includes extension metadata, so timestamp partials with different time units must not be
/// combined even if their storage dtypes match.
///
/// # Example
///
/// `vortex_aggregates::register_storage_ordered` registers the `is_sorted` and `min_max` aggregates
/// this way for the temporal extension types.
#[derive(Clone, Debug)]
pub struct ExtensionStoragePushdown<A: AggregateFnVTable, E: ExtVTable> {
    aggregate: A,
    ext: E,
    id: AggregateFnId,
}

impl<A: AggregateFnVTable, E: ExtVTable> ExtensionStoragePushdown<A, E> {
    /// Push `aggregate` down into the storage of extension type `ext`.
    ///
    /// Only do this when `aggregate` over the storage is correct for `ext`; see
    /// [`ExtensionStoragePushdown`] for how to check.
    pub fn new(aggregate: A, ext: E) -> Self {
        #[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
        let id = AggregateFnId::new(&format!("{}[{}]", aggregate.id(), ext.id()));
        Self { aggregate, ext, id }
    }

    /// The storage dtype of `dtype`, if it is an extension dtype of type `E`.
    fn storage_dtype<'a>(&self, dtype: &'a DType) -> Option<&'a DType> {
        dtype
            .as_extension_opt()
            .filter(|ext_dtype| ext_dtype.is::<E>())
            .map(|ext_dtype| ext_dtype.storage_dtype())
    }

    /// The arguments of the storage aggregate `A`.
    ///
    /// `A` shares this aggregate's options, return and partial dtypes, and its input dtype is
    /// borrowed from the extension dtype, so this allocates nothing.
    fn storage_args<'a>(
        &self,
        args: AggregateArgs<'a, A::Options>,
    ) -> VortexResult<AggregateArgs<'a, A::Options>> {
        let dtype = self.storage_dtype(args.dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} expects extension type {}, got {}",
                self.id,
                self.ext.id(),
                args.dtype
            )
        })?;
        Ok(AggregateArgs { dtype, ..args })
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

impl<A: AggregateFnVTable, E: ExtVTable> AggregateFnVTable for ExtensionStoragePushdown<A, E> {
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
        self.aggregate
            .return_dtype(options, self.storage_dtype(input_dtype)?)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.aggregate
            .partial_dtype(options, self.storage_dtype(input_dtype)?)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        self.aggregate.empty_partial(self.storage_args(args)?)
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        self.aggregate
            .partial_from_scalar(self.storage_args(args)?, scalar)
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        self.aggregate
            .merge_partials(self.storage_args(args)?, first, second)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.aggregate.to_scalar(self.storage_args(args)?, partial)
    }

    fn is_saturated(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        self.storage_args(args)
            .is_ok_and(|args| self.aggregate.is_saturated(args, partial))
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
        self.accumulate_storage(self.storage_args(args)?, state, array.storage_array(), ctx)?;
        Ok(true)
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let args = self.storage_args(args)?;
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
        self.aggregate.finalize(self.storage_args(args)?, states)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.aggregate
            .finalize_scalar(self.storage_args(args)?, partial)
    }
}
