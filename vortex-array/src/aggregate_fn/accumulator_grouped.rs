// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::ArrowNativeType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::arrays::ChunkedArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::vtable::ValidityHelper;

/// Reference-counted type-erased grouped accumulator.
pub type GroupedAccumulatorRef = Box<dyn DynGroupedAccumulator>;

/// An accumulator used for computing grouped aggregates.
///
/// Note that the groups must be processed in order, and the accumulator does not support random
/// access to groups.
pub struct GroupedAccumulator<V: AggregateFnVTable> {
    /// The vtable of the aggregate function.
    vtable: V,
    /// The options of the aggregate function.
    options: V::Options,
    /// Type-erased aggregate function used for kernel dispatch.
    aggregate_fn: AggregateFnRef,
    /// The DType of the input.
    dtype: DType,
    /// The DType of the aggregate.
    return_dtype: DType,
    /// The DType of the accumulator state.
    state_dtype: DType,
    /// The accumulated state for prior batches of groups.
    states: Vec<ArrayRef>,
    /// A session used to lookup custom aggregate kernels.
    session: VortexSession,
}

impl<V: AggregateFnVTable> GroupedAccumulator<V> {
    pub fn try_new(
        vtable: V,
        options: V::Options,
        dtype: DType,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let aggregate_fn = AggregateFn::new(vtable.clone(), options.clone()).erased();
        let return_dtype = vtable.return_dtype(&options, &dtype)?;
        let state_dtype = vtable.state_dtype(&options, &dtype)?;

        Ok(Self {
            vtable,
            options,
            aggregate_fn,
            dtype,
            return_dtype,
            state_dtype,
            states: vec![],
            session,
        })
    }
}

/// A trait object for type-erased grouped accumulators, used for dynamic dispatch when the aggregate
/// function is not known at compile time.
pub trait DynGroupedAccumulator: 'static + Send {
    /// Accumulate a list of groups into the accumulator.
    fn accumulate_list(&mut self, groups: &ArrayRef) -> VortexResult<()>;

    /// Finish the accumulation and return the final aggregate results for all groups.
    /// Resets the accumulator state for the next round of accumulation.
    fn finish(&mut self) -> VortexResult<ArrayRef>;
}

impl<V: AggregateFnVTable> DynGroupedAccumulator for GroupedAccumulator<V> {
    fn accumulate_list(&mut self, groups: &ArrayRef) -> VortexResult<()> {
        let elements_dtype = match groups.dtype() {
            DType::List(elem, _) => elem,
            DType::FixedSizeList(elem, ..) => elem,
            _ => vortex_bail!(
                "Input DType mismatch: expected List or FixedSizeList, got {}",
                groups.dtype()
            ),
        };
        vortex_ensure!(
            elements_dtype.as_ref() == &self.dtype,
            "Input DType mismatch: expected {}, got {}",
            self.dtype,
            elements_dtype
        );

        let mut ctx = self.session.create_execution_ctx();

        // We first execute the groups until it is a ListView or FixedSizeList, since we only
        // dispatch the aggregate kernel over the elements of these arrays.
        match groups.execute::<Canonical>(&mut ctx)? {
            Canonical::List(groups) => self.accumulate_list_view(&groups, &mut ctx),
            Canonical::FixedSizeList(groups) => self.accumulate_fixed_size_list(&groups, &mut ctx),
            _ => vortex_panic!("We checked the DType above, so this should never happen"),
        }
    }

    fn finish(&mut self) -> VortexResult<ArrayRef> {
        let states = std::mem::replace(&mut self.states, vec![]);
        Ok(ChunkedArray::try_new(states, self.state_dtype.clone())?.into_array())
    }
}

impl<V: AggregateFnVTable> GroupedAccumulator<V> {
    fn accumulate_list_view(
        &mut self,
        groups: &ListViewArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut elements = groups.elements().clone();

        let kernels = self.session.aggregate_fns().grouped_kernels;

        for _ in 0..64 {
            if elements.is::<AnyCanonical>() {
                break;
            }

            let kernel_key = (self.vtable.id(), elements.encoding_id());
            if let Some(kernel) = kernels.get(&kernel_key) {
                // SAFETY: we assume that elements execution is safe
                let groups = unsafe {
                    ListViewArray::new_unchecked(
                        elements.clone(),
                        groups.offsets().clone(),
                        groups.sizes().clone(),
                        groups.validity().clone(),
                    )
                };

                if let Some(result) = kernel.grouped_aggregate(&self.aggregate_fn, &groups)? {
                    return self.push_result(result);
                }
            }

            // Execute one step and try again
            elements = elements.execute(ctx)?;
        }

        // Otherwise, we iterate the offsets and sizes and accumulate each group one by one.
        let elements = elements.execute::<Canonical>(ctx)?;
        let offsets = groups.offsets();
        let sizes = groups.sizes().cast(offsets.dtype().clone())?;
        let validity = groups.validity().to_mask(offsets.len());

        match_each_integer_ptype!(offsets.ptype(), |O| {
            let offsets = offsets.execute::<Buffer<O>>(ctx)?;
            let sizes = sizes.execute::<Buffer<O>>(ctx)?;
            self.accumulate_list_view_typed(&elements, offsets.as_ref(), sizes.as_ref(), validity)
        })
    }

    fn accumulate_list_view_typed<O: IntegerPType>(
        &mut self,
        elements: &ArrayRef,
        offsets: &[O],
        sizes: &[O],
        validity: &Mask,
    ) -> VortexResult<()> {
        let mut accumulator = Accumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            self.dtype.clone(),
            self.session.clone(),
        )?;
        let mut states = builder_with_capacity(&self.state_dtype, offsets.len());

        for (offset, size) in offsets.iter().zip(sizes.iter()) {
            let offset = offset.to_usize().vortex_expect("Offset value is not usize");
            let size = size.to_usize().vortex_expect("Size value is not usize");

            if validity.value(offset) {
                let group = elements.slice(offset..offset + size)?;
                accumulator.accumulate(&group)?;
                states.append_scalar(&accumulator.finish())?;
                accumulator.reset();
            } else {
                states.append_null()
            }
        }

        self.push_result(states.finish())
    }

    fn accumulate_fixed_size_list(
        &mut self,
        groups: &FixedSizeListArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut elements = groups.elements().clone();

        let kernels = self.session.aggregate_fns().grouped_kernels;

        for _ in 0..64 {
            if elements.is::<AnyCanonical>() {
                break;
            }

            let kernel_key = (self.vtable.id(), elements.encoding_id());
            if let Some(kernel) = kernels.get(&kernel_key) {
                // SAFETY: we assume that elements execution is safe
                let groups = unsafe {
                    FixedSizeListArray::new_unchecked(
                        elements.clone(),
                        groups.list_size(),
                        groups.validity().clone(),
                        groups.len(),
                    )
                };

                if let Some(result) =
                    kernel.grouped_aggregate_fixed_size(&self.aggregate_fn, &groups)?
                {
                    return self.push_result(result);
                }
            }

            // Execute one step and try again
            elements = elements.execute(ctx)?;
        }

        // Otherwise, execute the batch until it is canonical and accumulate it into the state.
        let canonical = elements.execute::<Canonical>(ctx)?;
        // SAFETY: we assume that elements execution is safe
        let groups = unsafe {
            FixedSizeListArray::new_unchecked(
                canonical.into_array(),
                groups.list_size(),
                groups.validity().clone(),
                groups.len(),
            )
        };

        self.push_result(self.vtable.accumulate_groups(&groups, ctx)?)
    }

    fn accumulate_fixed_size_list_typed(
        &mut self,
        groups: &FixedSizeListArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let validity = groups.validity().to_mask(groups.len());

        let mut accumulator = Accumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            self.dtype.clone(),
            self.session.clone(),
        )?;
        let mut states = builder_with_capacity(&self.state_dtype, groups.len());

        let mut offset = 0;
        let size = groups
            .list_size()
            .to_usize()
            .vortex_expect("List size is not usize");

        for i in 0..groups.len() {
            if validity.value(i) {
                let group = groups.elements().slice(offset..offset + size)?;
                accumulator.accumulate(&group)?;
                states.append_scalar(&accumulator.finish())?;
            } else {
                states.append_null()
            }
            offset += size;
        }

        self.push_result(states.finish())
    }

    fn push_result(&mut self, state: ArrayRef) -> VortexResult<()> {
        vortex_ensure!(
            state.dtype() == &self.state_dtype,
            "State DType mismatch: expected {}, got {}",
            self.state_dtype,
            state.dtype()
        );
        self.states.push(state);
        Ok(())
    }
}
