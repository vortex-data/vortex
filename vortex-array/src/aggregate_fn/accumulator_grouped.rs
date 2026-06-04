// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::ArrowNativeType;
use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::AnyCanonical;
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
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::executor::max_iterations;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;

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
            Canonical::List(groups) => self.accumulate_list_view(&groups, ctx),
            Canonical::FixedSizeList(groups) => self.accumulate_fixed_size_list(&groups, ctx),
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
    fn accumulate_list_view(
        &mut self,
        groups: &ListViewArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut elements = groups.elements().clone();
        let groups_validity = groups.validity()?;
        let session = ctx.session().clone();
        let kernels = &session.aggregate_fns().grouped_kernels;

        for _ in 0..max_iterations() {
            if elements.is::<AnyCanonical>() {
                break;
            }

            let kernels_r = kernels.load();
            if let Some(result) = kernels_r
                .get(&(elements.encoding_id(), Some(self.aggregate_fn.id())))
                .or_else(|| kernels_r.get(&(elements.encoding_id(), None)))
                .and_then(|kernel| {
                    // SAFETY: we assume that elements execution is safe
                    let groups = unsafe {
                        ListViewArray::new_unchecked(
                            elements.clone(),
                            groups.offsets().clone(),
                            groups.sizes().clone(),
                            groups_validity.clone(),
                        )
                    };
                    kernel
                        .grouped_aggregate(&self.aggregate_fn, &groups)
                        .transpose()
                })
                .transpose()?
            {
                return self.push_result(result);
            }

            // Execute one step and try again
            elements = elements.execute(ctx)?;
        }

        // Otherwise, we iterate the offsets and sizes and accumulate each group one by one.
        let elements = elements.execute::<Columnar>(ctx)?.into_array();
        let offsets = groups.offsets();
        let sizes = groups.sizes().cast(offsets.dtype().clone())?;
        let validity = groups_validity.execute_mask(offsets.len(), ctx)?;

        match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
            let offsets = offsets.clone().execute::<Buffer<O>>(ctx)?;
            let sizes = sizes.execute::<Buffer<O>>(ctx)?;
            self.accumulate_list_view_typed(
                &elements,
                offsets.as_ref(),
                sizes.as_ref(),
                &validity,
                ctx,
            )
        })
    }

    fn accumulate_list_view_typed<O: IntegerPType>(
        &mut self,
        elements: &ArrayRef,
        offsets: &[O],
        sizes: &[O],
        validity: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Fast path: summing a canonical, all-valid primitive column. The generic loop below
        // creates a slice array + does a kernel-dispatching scalar `accumulate` + scalar boxing per
        // group, which is catastrophic for many small groups. A direct typed per-group slice sum is
        // orders of magnitude faster. Falls through for null elements / other aggregates.
        if self.aggregate_fn.id().as_str() == "vortex.sum"
            && let Some(result) = try_sum_primitive_groups(
                elements,
                offsets,
                sizes,
                validity,
                &self.partial_dtype,
                ctx,
            )?
        {
            return self.push_result(result);
        }

        let mut accumulator = Accumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            self.dtype.clone(),
        )?;
        let mut states = builder_with_capacity(&self.partial_dtype, offsets.len());

        for (offset, size) in offsets.iter().zip(sizes.iter()) {
            let offset = offset.to_usize().vortex_expect("Offset value is not usize");
            let size = size.to_usize().vortex_expect("Size value is not usize");

            if validity.value(offset) {
                let group = elements.slice(offset..offset + size)?;
                accumulator.accumulate(&group, ctx)?;
                states.append_scalar(&accumulator.flush()?)?;
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
        let groups_validity = groups.validity()?;
        let session = ctx.session().clone();
        let kernels = &session.aggregate_fns().grouped_kernels;

        for _ in 0..64 {
            if elements.is::<AnyCanonical>() {
                break;
            }

            let kernels_r = kernels.load();
            if let Some(result) = kernels_r
                .get(&(elements.encoding_id(), Some(self.aggregate_fn.id())))
                .or_else(|| kernels_r.get(&(elements.encoding_id(), None)))
                .and_then(|kernel| {
                    // SAFETY: we assume that elements execution is safe
                    let groups = unsafe {
                        FixedSizeListArray::new_unchecked(
                            elements.clone(),
                            groups.list_size(),
                            groups_validity.clone(),
                            groups.len(),
                        )
                    };

                    kernel
                        .grouped_aggregate_fixed_size(&self.aggregate_fn, &groups)
                        .transpose()
                })
                .transpose()?
            {
                return self.push_result(result);
            }

            // Execute one step and try again
            elements = elements.execute(ctx)?;
        }

        // Otherwise, we iterate the offsets and sizes and accumulate each group one by one.
        let elements = elements.execute::<Columnar>(ctx)?.into_array();
        let validity = groups_validity.execute_mask(groups.len(), ctx)?;

        let mut accumulator = Accumulator::try_new(
            self.vtable.clone(),
            self.options.clone(),
            self.dtype.clone(),
        )?;
        let mut states = builder_with_capacity(&self.partial_dtype, groups.len());

        let mut offset = 0;
        let size = groups
            .list_size()
            .to_usize()
            .vortex_expect("List size is not usize");

        for i in 0..groups.len() {
            if validity.value(i) {
                let group = elements.slice(offset..offset + size)?;
                accumulator.accumulate(&group, ctx)?;
                states.append_scalar(&accumulator.flush()?)?;
            } else {
                states.append_null()
            }
            offset += size;
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

/// Fast vectorized per-group `Sum` over a canonical, all-valid primitive elements array.
///
/// Returns `Ok(None)` (caller falls back to the generic per-group path) when the elements are not a
/// canonical primitive, contain nulls, or the result dtype would not match the `Sum` partial dtype.
fn try_sum_primitive_groups<O: IntegerPType>(
    elements: &ArrayRef,
    offsets: &[O],
    sizes: &[O],
    group_validity: &Mask,
    partial_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(prim) = elements.as_opt::<Primitive>() else {
        return Ok(None);
    };
    // Materialize the element validity once. The common case (a nullable column with no actual
    // nulls) is `AllOr::All` and takes the tight slice-sum loop; mixed validity falls to a masked
    // loop. (`AllOr::None` -> every valid group sums to zero, matching `Sum`.)
    let elem_mask = prim.validity()?.execute_mask(prim.len(), ctx)?;
    let all_valid = matches!(elem_mask.slices(), AllOr::All);

    let result = match_each_native_ptype!(prim.ptype(),
        unsigned: |T| { sum_groups_unsigned::<T, O>(prim.as_slice::<T>(), offsets, sizes, group_validity, &elem_mask, all_valid) },
        signed:   |T| { sum_groups_signed::<T, O>(prim.as_slice::<T>(), offsets, sizes, group_validity, &elem_mask, all_valid) },
        floating: |T| { sum_groups_float::<T, O>(prim.as_slice::<T>(), offsets, sizes, group_validity, &elem_mask, all_valid) }
    );

    // Defensive: if our widening doesn't exactly match the aggregate's partial dtype, fall back to
    // the generic path rather than emit a mistyped array downstream.
    if result.dtype() != partial_dtype {
        return Ok(None);
    }
    Ok(Some(result))
}

fn sum_groups_unsigned<T, O>(
    values: &[T],
    offsets: &[O],
    sizes: &[O],
    group_validity: &Mask,
    elem_mask: &Mask,
    all_valid: bool,
) -> ArrayRef
where
    T: NativePType + AsPrimitive<u64>,
    O: IntegerPType,
{
    let iter = offsets
        .iter()
        .zip(sizes.iter())
        .enumerate()
        .map(|(i, (o, sz))| {
            if !group_validity.value(i) {
                return None;
            }
            let o = o.to_usize().vortex_expect("offset usize");
            let sz = sz.to_usize().vortex_expect("size usize");
            let mut acc: u64 = 0;
            if all_valid {
                for &v in &values[o..o + sz] {
                    acc = acc.checked_add(v.as_())?; // overflow -> null, matching Sum saturation
                }
            } else {
                for j in 0..sz {
                    if elem_mask.value(o + j) {
                        acc = acc.checked_add(values[o + j].as_())?;
                    }
                }
            }
            Some(acc)
        });
    PrimitiveArray::from_option_iter(iter).into_array()
}

fn sum_groups_signed<T, O>(
    values: &[T],
    offsets: &[O],
    sizes: &[O],
    group_validity: &Mask,
    elem_mask: &Mask,
    all_valid: bool,
) -> ArrayRef
where
    T: NativePType + AsPrimitive<i64>,
    O: IntegerPType,
{
    let iter = offsets
        .iter()
        .zip(sizes.iter())
        .enumerate()
        .map(|(i, (o, sz))| {
            if !group_validity.value(i) {
                return None;
            }
            let o = o.to_usize().vortex_expect("offset usize");
            let sz = sz.to_usize().vortex_expect("size usize");
            let mut acc: i64 = 0;
            if all_valid {
                for &v in &values[o..o + sz] {
                    acc = acc.checked_add(v.as_())?; // overflow -> null, matching Sum saturation
                }
            } else {
                for j in 0..sz {
                    if elem_mask.value(o + j) {
                        acc = acc.checked_add(values[o + j].as_())?;
                    }
                }
            }
            Some(acc)
        });
    PrimitiveArray::from_option_iter(iter).into_array()
}

fn sum_groups_float<T, O>(
    values: &[T],
    offsets: &[O],
    sizes: &[O],
    group_validity: &Mask,
    elem_mask: &Mask,
    all_valid: bool,
) -> ArrayRef
where
    T: NativePType + ToPrimitive,
    O: IntegerPType,
{
    let iter = offsets
        .iter()
        .zip(sizes.iter())
        .enumerate()
        .map(|(i, (o, sz))| {
            if !group_validity.value(i) {
                return None;
            }
            let o = o.to_usize().vortex_expect("offset usize");
            let sz = sz.to_usize().vortex_expect("size usize");
            let mut acc: f64 = 0.0;
            // NaN propagates, matching Sum's float semantics.
            if all_valid {
                for &v in &values[o..o + sz] {
                    acc += v.to_f64().vortex_expect("float to f64");
                }
            } else {
                for j in 0..sz {
                    if elem_mask.value(o + j) {
                        acc += values[o + j].to_f64().vortex_expect("float to f64");
                    }
                }
            }
            Some(acc)
        });
    PrimitiveArray::from_option_iter(iter).into_array()
}
