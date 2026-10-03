// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pluggable aggregate function kernels used to provide encoding-specific implementations of
//! aggregate functions.

use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::GroupedArray;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// A pluggable kernel for an aggregate function.
///
/// The provided array should be aggregated into a single scalar representing the partial state
/// of a single group.
pub trait DynAggregateKernel: 'static + Send + Sync + Debug {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>>;
}

/// Aggregates an extension-typed batch as the aggregate `A` of its storage values.
///
/// Registering this kernel for an extension type with
/// [`register_extension_aggregate_kernel`](crate::aggregate_fn::session::AggregateFnSession::register_extension_aggregate_kernel)
/// declares that, for that type, `A` agrees with `A` over the storage values, for example that the
/// type sorts as its storage does. It declines when `A`'s partial state differs between the
/// extension dtype and its storage dtype.
pub struct StorageAggregateKernel<A>(PhantomData<fn() -> A>);

impl<A> StorageAggregateKernel<A> {
    /// The kernel instance, for registration as a `&'static` kernel.
    pub const NEW: Self = Self(PhantomData);
}

impl<A> Debug for StorageAggregateKernel<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StorageAggregateKernel<{}>", std::any::type_name::<A>())
    }
}

impl<A: AggregateFnVTable> DynAggregateKernel for StorageAggregateKernel<A> {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if aggregate_fn.as_opt::<A>().is_none() {
            return Ok(None);
        }
        let DType::Extension(ext_dtype) = batch.dtype() else {
            return Ok(None);
        };
        let storage_dtype = ext_dtype.storage_dtype();
        if aggregate_fn.state_dtype(batch.dtype()) != aggregate_fn.state_dtype(storage_dtype) {
            return Ok(None);
        }

        let storage = batch
            .clone()
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone();
        let mut accumulator = aggregate_fn.accumulator(storage_dtype)?;
        accumulator.accumulate(&storage, ctx)?;
        accumulator.flush().map(Some)
    }
}

/// A pluggable kernel for batch aggregation of many groups.
///
/// A kernel can be registered either for an aggregate function regardless of the element encoding,
/// or for a specific aggregate function and element encoding. Element-encoding kernels are matched
/// on the inner array of the provided grouped array, not on the outer list encoding. This is more
/// pragmatic than having every kernel match on the outer list encoding and having to deal with the
/// possibility of multiple list encodings.
///
/// Each value in the grouped array represents a group and the result of the grouped aggregate
/// should be an array of the same length, where each element is the aggregate state of the
/// corresponding group.
///
/// Return `Ok(None)` if the kernel cannot be applied to the given aggregate function.
pub trait DynGroupedAggregateKernel: 'static + Send + Sync + Debug {
    /// Aggregate each group in the provided grouped array and return an array of the aggregate
    /// states.
    fn grouped_aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        groups: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}
