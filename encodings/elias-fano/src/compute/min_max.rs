// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::make_minmax_dtype;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::EliasFano;
use crate::EliasFanoCursor;

/// Elias-Fano-specific min/max kernel: the sequence is sorted, so two `select1`s find the extremes.
///
/// Deliberately not read from the `reference` and `max` metadata, which describe the *encoded*
/// universe and survive slicing, so on a sliced array they are not its extremes.
///
/// Each end goes through a one-element slice rather than one cursor over the whole array: a cursor
/// sizes its low-bits view to the array it opens on, and materialises the slot whole if it cannot
/// be read in place.
#[derive(Debug)]
pub(crate) struct EliasFanoMinMaxKernel;

impl DynAggregateKernel for EliasFanoMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }
        let Some(array) = batch.as_opt::<EliasFano>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());
        if array.is_empty() {
            return Ok(Some(Scalar::null(struct_dtype)));
        }

        let last = array.len() - 1;
        let min = element_at(batch, 0, ctx)?;
        let max = element_at(batch, last, ctx)?;

        Ok(Some(Scalar::struct_(struct_dtype, vec![min, max])))
    }
}

fn element_at(array: &ArrayRef, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let one = array.slice(index..index + 1)?;
    let Some(one) = one.as_opt::<EliasFano>() else {
        // Slicing normally reduces into the encoding, but it is not obliged to.
        return one.execute_scalar(0, ctx);
    };
    EliasFanoCursor::try_new(one, ctx)?.access(0)
}
