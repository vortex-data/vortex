// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::ArrowNativeType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::session::AggregateFnSessionExt;
use crate::arrays::ChunkedArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::columnar::AnyColumnar;
use crate::dtype::DType;
use crate::executor::max_iterations;
use crate::match_each_integer_ptype;

/// Reference-counted type-erased grouped accumulator.
pub type GroupedAccumulatorRef = Box<dyn DynGroupedAccumulator>;

/// A batch of grouped values to aggregate.
///
/// Each outer list value is one group, and the inner element array is shared by all groups.
/// Aggregate implementations can inspect the concrete grouped representation directly, or ask for
/// derived ranges when their algorithm is expressed in terms of `(offset, size)` pairs.
pub enum GroupedArray {
    /// Groups represented as a list-view array with per-group offsets and sizes.
    ListView(ListViewArray),
    /// Groups represented as a fixed-size list array.
    FixedSizeList(FixedSizeListArray),
}

impl From<ListViewArray> for GroupedArray {
    fn from(groups: ListViewArray) -> Self {
        Self::ListView(groups)
    }
}

impl From<FixedSizeListArray> for GroupedArray {
    fn from(groups: FixedSizeListArray) -> Self {
        Self::FixedSizeList(groups)
    }
}

impl GroupedArray {
    /// The inner element array shared by all groups.
    pub fn elements(&self) -> &ArrayRef {
        match self {
            Self::ListView(groups) => groups.elements(),
            Self::FixedSizeList(groups) => groups.elements(),
        }
    }

    /// Return the `(offset, size)` ranges describing each group in `elements`.
    pub fn group_ranges(&self, ctx: &mut ExecutionCtx) -> VortexResult<GroupRanges> {
        match self {
            Self::ListView(groups) => list_view_group_ranges(groups, ctx),
            Self::FixedSizeList(groups) => Ok(fixed_size_list_group_ranges(groups)),
        }
    }

    /// Return the per-group validity mask.
    pub fn group_validity(&self, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        match self {
            Self::ListView(groups) => groups.validity()?.execute_mask(groups.len(), ctx),
            Self::FixedSizeList(groups) => groups.validity()?.execute_mask(groups.len(), ctx),
        }
    }

    /// The number of groups in this batch.
    pub fn len(&self) -> usize {
        match self {
            Self::ListView(groups) => groups.len(),
            Self::FixedSizeList(groups) => groups.len(),
        }
    }

    /// Returns true when this batch contains no groups.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true when every group is valid.
    pub fn all_groups_valid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(self.group_validity(ctx)?.all_true())
    }

    unsafe fn with_elements_unchecked(&self, elements: ArrayRef) -> VortexResult<Self> {
        Ok(match self {
            Self::ListView(groups) => unsafe {
                ListViewArray::new_unchecked(
                    elements,
                    groups.offsets().clone(),
                    groups.sizes().clone(),
                    groups.validity()?,
                )
            }
            .into(),
            Self::FixedSizeList(groups) => unsafe {
                FixedSizeListArray::new_unchecked(
                    elements,
                    groups.list_size(),
                    groups.validity()?,
                    groups.len(),
                )
            }
            .into(),
        })
    }
}

/// The physical ranges of a grouped array.
pub enum GroupRanges {
    /// Explicit ranges extracted from a list-view array.
    ListView {
        /// The `(offset, size)` ranges.
        ranges: Vec<(usize, usize)>,
    },
    /// Uniform ranges derived from a fixed-size list array.
    FixedSizeList {
        /// The number of groups.
        len: usize,
        /// The number of elements in each group.
        size: usize,
    },
}

impl GroupRanges {
    /// The number of groups described by these ranges.
    pub fn len(&self) -> usize {
        match self {
            Self::ListView { ranges } => ranges.len(),
            Self::FixedSizeList { len, .. } => *len,
        }
    }

    /// Returns true when there are no groups.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the `(offset, size)` range for the group at `index`.
    fn range(&self, index: usize) -> (usize, usize) {
        match self {
            Self::ListView { ranges } => ranges[index],
            Self::FixedSizeList { len, size } => {
                assert!(index < *len, "range index out of bounds");
                (index * size, *size)
            }
        }
    }

    /// Iterate over all `(offset, size)` group ranges.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.len()).map(|index| self.range(index))
    }
}

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
    /// The DType of the partial accumulator state.
    partial_dtype: DType,
    /// The accumulated state for prior batches of groups.
    partials: Vec<ArrayRef>,
}

impl<V: AggregateFnVTable> GroupedAccumulator<V> {
    pub fn try_new(vtable: V, options: V::Options, dtype: DType) -> VortexResult<Self> {
        let aggregate_fn = AggregateFn::new(vtable.clone(), options.clone()).erased();
        let return_dtype = vtable.return_dtype(&options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;
        let partial_dtype = vtable.partial_dtype(&options, &dtype).ok_or_else(|| {
            vortex_err!(
                "Aggregate function {} cannot be applied to dtype {}",
                vtable.id(),
                dtype
            )
        })?;

        Ok(Self {
            vtable,
            options,
            aggregate_fn,
            dtype,
            return_dtype,
            partial_dtype,
            partials: vec![],
        })
    }
}

/// A trait object for type-erased grouped accumulators, used for dynamic dispatch when the aggregate
/// function is not known at compile time.
pub trait DynGroupedAccumulator: 'static + Send {
    /// Accumulate a list of groups into the accumulator.
    fn accumulate_list(&mut self, groups: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()>;

    /// Finish the accumulation and return the partial aggregate results for all groups.
    /// Resets the accumulator state for the next round of accumulation.
    fn flush(&mut self) -> VortexResult<ArrayRef>;

    /// Finish the accumulation and return the final aggregate results for all groups.
    /// Resets the accumulator state for the next round of accumulation.
    fn finish(&mut self) -> VortexResult<ArrayRef>;
}

impl<V: AggregateFnVTable> DynGroupedAccumulator for GroupedAccumulator<V> {
    fn accumulate_list(&mut self, groups: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
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

        // We first execute the groups until it is a ListView or FixedSizeList, since we only
        // dispatch the aggregate kernel over the elements of these arrays.
        let canonical = match groups.clone().execute::<Columnar>(ctx)? {
            Columnar::Canonical(c) => c,
            Columnar::Constant(c) => c.into_array().execute::<Canonical>(ctx)?,
        };
        match canonical {
            Canonical::List(groups) => self.accumulate_grouped_array(groups.into(), ctx),
            Canonical::FixedSizeList(groups) => self.accumulate_grouped_array(groups.into(), ctx),
            _ => vortex_panic!("We checked the DType above, so this should never happen"),
        }
    }

    fn flush(&mut self) -> VortexResult<ArrayRef> {
        let states = std::mem::take(&mut self.partials);
        Ok(ChunkedArray::try_new(states, self.partial_dtype.clone())?.into_array())
    }

    fn finish(&mut self) -> VortexResult<ArrayRef> {
        let states = self.flush()?;
        let results = self.vtable.finalize(states)?;

        vortex_ensure!(
            results.dtype() == &self.return_dtype,
            "Return DType mismatch: expected {}, got {}",
            self.return_dtype,
            results.dtype()
        );

        Ok(results)
    }
}

impl<V: AggregateFnVTable> GroupedAccumulator<V> {
    fn accumulate_grouped_array(
        &mut self,
        groups: GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut elements = groups.elements().clone();
        let session = ctx.session().clone();

        for _ in 0..max_iterations() {
            // Try a registered grouped kernel for the current element encoding.
            if let Some(kernel) = session
                .aggregate_fns()
                .find_grouped_encoding_kernel(elements.encoding_id(), self.aggregate_fn.id())
            {
                // SAFETY: we assume that elements execution is safe
                let kernel_groups = unsafe { groups.with_elements_unchecked(elements.clone())? };
                if let Some(result) =
                    kernel.grouped_aggregate(&self.aggregate_fn, &kernel_groups, ctx)?
                {
                    return self.push_result(result);
                }
            }

            // Try a grouped kernel for the current aggregate regardless of element encoding.
            if let Some(kernel) = session
                .aggregate_fns()
                .find_grouped_kernel(self.aggregate_fn.id())
            {
                // SAFETY: we preserve the grouped shape and validity while replacing the
                // elements with another representation of the same logical array.
                let kernel_groups = unsafe { groups.with_elements_unchecked(elements.clone())? };
                if let Some(result) =
                    kernel.grouped_aggregate(&self.aggregate_fn, &kernel_groups, ctx)?
                {
                    return self.push_result(result);
                }
            }

            if elements.is::<AnyColumnar>() {
                break;
            }

            // Execute one step and try again
            elements = elements.execute(ctx)?;
        }

        let elements = elements.execute::<Columnar>(ctx)?.into_array();
        // SAFETY: we preserve the grouped shape and validity while replacing the elements with an
        // executed form of the same logical array.
        let grouped = unsafe { groups.with_elements_unchecked(elements)? };

        // Otherwise, we iterate the offsets and sizes and accumulate each group one by one.
        self.accumulate_grouped_fallback(&grouped, ctx)
    }

    fn accumulate_grouped_fallback(
        &mut self,
        grouped: &GroupedArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut accumulator = Accumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            self.dtype.clone(),
        )?;
        let mut states = builder_with_capacity(&self.partial_dtype, grouped.len());
        let group_ranges = grouped.group_ranges(ctx)?;
        let group_validity = grouped.group_validity(ctx)?;

        for ((offset, size), valid) in group_ranges.iter().zip(group_validity.iter()) {
            if valid {
                let group = grouped.elements().slice(offset..offset + size)?;
                accumulator.accumulate(&group, ctx)?;
                states.append_scalar(&accumulator.flush()?)?;
            } else {
                states.append_null()
            }
        }

        self.push_result(states.finish())
    }

    fn push_result(&mut self, state: ArrayRef) -> VortexResult<()> {
        vortex_ensure!(
            state.dtype() == &self.partial_dtype,
            "State DType mismatch: expected {}, got {}",
            self.partial_dtype,
            state.dtype()
        );
        self.partials.push(state);
        Ok(())
    }
}
fn list_view_group_ranges(
    groups: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GroupRanges> {
    let offsets = groups.offsets();
    let sizes = groups.sizes().cast(offsets.dtype().clone())?;

    let ranges = match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
        let offsets = offsets.clone().execute::<Buffer<O>>(ctx)?;
        let sizes = sizes.execute::<Buffer<O>>(ctx)?;
        offsets
            .as_ref()
            .iter()
            .zip(sizes.as_ref().iter())
            .map(|(offset, size)| {
                (
                    offset.to_usize().vortex_expect("Offset value is not usize"),
                    size.to_usize().vortex_expect("Size value is not usize"),
                )
            })
            .collect::<Vec<_>>()
    });

    Ok(GroupRanges::ListView { ranges })
}

fn fixed_size_list_group_ranges(groups: &FixedSizeListArray) -> GroupRanges {
    GroupRanges::FixedSizeList {
        len: groups.len(),
        size: groups.list_size() as usize,
    }
}
