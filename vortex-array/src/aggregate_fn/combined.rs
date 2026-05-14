// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic adapter for aggregates whose result is computed from two child
//! aggregate functions, e.g. `Mean = Sum / Count`.

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::{self};
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AccumulatorRef;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::scalar::Scalar;

/// Pair of options for the two children of a [`BinaryCombined`] aggregate.
///
/// Wrapper around `(L, R)` because the [`AggregateFnVTable::Options`] bound
/// requires `Display`, which tuples don't implement.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PairOptions<L, R>(pub L, pub R);

impl<L: Display, R: Display> Display for PairOptions<L, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

// Convenience aliases so signatures stay readable.
type LeftOptions<T> = <<T as BinaryCombined>::Left as AggregateFnVTable>::Options;
type RightOptions<T> = <<T as BinaryCombined>::Right as AggregateFnVTable>::Options;
/// Combined options for a [`BinaryCombined`] aggregate.
pub type CombinedOptions<T> = PairOptions<LeftOptions<T>, RightOptions<T>>;

/// Declare an aggregate function in terms of two child aggregates.
pub trait BinaryCombined: 'static + Send + Sync + Clone {
    /// The left child aggregate vtable.
    type Left: AggregateFnVTable;
    /// The right child aggregate vtable.
    type Right: AggregateFnVTable;

    /// Stable identifier for the combined aggregate.
    fn id(&self) -> AggregateFnId;

    /// Construct the left child vtable.
    fn left(&self) -> Self::Left;

    /// Construct the right child vtable.
    fn right(&self) -> Self::Right;

    /// Field name for the left child in the partial struct dtype.
    fn left_name(&self) -> &'static str {
        "left"
    }

    /// Field name for the right child in the partial struct dtype.
    fn right_name(&self) -> &'static str {
        "right"
    }

    /// Return type of the combined aggregate.
    fn return_dtype(&self, input_dtype: &DType) -> Option<DType>;

    /// Combine the finalized left and right results into the final aggregate.
    fn finalize(&self, left: ArrayRef, right: ArrayRef) -> VortexResult<ArrayRef>;

    fn finalize_scalar(&self, left_scalar: Scalar, right_scalar: Scalar) -> VortexResult<Scalar>;

    /// Serialize the options for this combined aggregate. Default: not serializable.
    fn serialize(&self, options: &CombinedOptions<Self>) -> VortexResult<Option<Vec<u8>>> {
        let _ = options;
        Ok(None)
    }

    /// Deserialize the options for this combined aggregate. Default: bails.
    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<CombinedOptions<Self>> {
        let _ = (metadata, session);
        vortex_bail!(
            "Combined aggregate function {} is not deserializable",
            BinaryCombined::id(self)
        );
    }

    /// Coerce the input type. Default: chains `right.coerce_args(left.coerce_args(input))`.
    fn coerce_args(
        &self,
        options: &CombinedOptions<Self>,
        input_dtype: &DType,
    ) -> VortexResult<DType> {
        let left_coerced = self.left().coerce_args(&options.0, input_dtype)?;
        self.right().coerce_args(&options.1, &left_coerced)
    }

    /// Build the partial struct dtype that wraps the two child partials.
    fn partial_struct_dtype(&self, left: DType, right: DType) -> DType {
        DType::Struct(
            StructFields::new(
                FieldNames::from_iter([
                    FieldName::from(self.left_name()),
                    FieldName::from(self.right_name()),
                ]),
                vec![left, right],
            ),
            Nullability::NonNullable,
        )
    }
}

/// Adapter that exposes any [`BinaryCombined`] as an [`AggregateFnVTable`].
#[derive(Clone, Debug)]
pub struct Combined<T: BinaryCombined>(pub T);

impl<T: BinaryCombined> Combined<T> {
    /// Construct a new combined aggregate vtable.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: BinaryCombined> AggregateFnVTable for Combined<T> {
    type Options = CombinedOptions<T>;
    // Each child is held as a fully-fledged `AccumulatorRef` so that batches dispatched through
    // `try_accumulate` consult the kernel registry per-child (e.g. a `(Dict, Sum)` kernel fires
    // for the inner `Sum` child of `Combined<Mean>`).
    type Partial = (AccumulatorRef, AccumulatorRef);

    fn id(&self) -> AggregateFnId {
        self.0.id()
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        BinaryCombined::serialize(&self.0, options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        BinaryCombined::deserialize(&self.0, metadata, session)
    }

    fn coerce_args(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        BinaryCombined::coerce_args(&self.0, options, input_dtype)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        BinaryCombined::return_dtype(&self.0, input_dtype)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let l = self.0.left().partial_dtype(&options.0, input_dtype)?;
        let r = self.0.right().partial_dtype(&options.1, input_dtype)?;
        Some(self.0.partial_struct_dtype(l, r))
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        let left = Accumulator::try_new(self.0.left(), options.0.clone(), input_dtype.clone())?;
        let right = Accumulator::try_new(self.0.right(), options.1.clone(), input_dtype.clone())?;
        Ok((
            Box::new(left) as AccumulatorRef,
            Box::new(right) as AccumulatorRef,
        ))
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }
        let s = other.as_struct();
        let lname = self.0.left_name();
        let rname = self.0.right_name();
        let l_field = s
            .field(lname)
            .ok_or_else(|| vortex_err!("BinaryCombined partial missing `{}` field", lname))?;
        let r_field = s
            .field(rname)
            .ok_or_else(|| vortex_err!("BinaryCombined partial missing `{}` field", rname))?;
        partial.0.combine_partials(l_field)?;
        partial.1.combine_partials(r_field)?;
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let l_scalar = partial.0.partial_scalar()?;
        let r_scalar = partial.1.partial_scalar()?;
        let dtype = self
            .0
            .partial_struct_dtype(l_scalar.dtype().clone(), r_scalar.dtype().clone());
        Ok(Scalar::struct_(dtype, vec![l_scalar, r_scalar]))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.0.reset();
        partial.1.reset();
    }

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.0.is_saturated() && partial.1.is_saturated()
    }

    /// Delegate the batch to each child's `Accumulator::accumulate`, which consults the
    /// kernel registry against the child's `aggregate_fn` id. This is what makes
    /// `(encoding, Child)` kernels reachable through `Combined<Parent>` — without it, a
    /// `(Dict, Sum)` kernel would be dead code for `Combined<Mean>`. We always return
    /// `true` so [`Self::accumulate`] is unreachable.
    fn try_accumulate(
        &self,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        state.0.accumulate(batch, ctx)?;
        state.1.accumulate(batch, ctx)?;
        Ok(true)
    }

    fn accumulate(
        &self,
        _state: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        unreachable!("Combined::try_accumulate handles all batches")
    }

    fn finalize(&self, states: ArrayRef) -> VortexResult<ArrayRef> {
        let l_field = states.get_item(FieldName::from(self.0.left_name()))?;
        let r_field = states.get_item(FieldName::from(self.0.right_name()))?;
        let l_finalized = self.0.left().finalize(l_field)?;
        let r_finalized = self.0.right().finalize(r_field)?;
        BinaryCombined::finalize(&self.0, l_finalized, r_finalized)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let l_scalar = partial.0.final_scalar()?;
        let r_scalar = partial.1.final_scalar()?;
        BinaryCombined::finalize_scalar(&self.0, l_scalar, r_scalar)
    }
}
