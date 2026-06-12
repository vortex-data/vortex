// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::SumPartial;
use super::SumState;
use super::checked_add_i64;
use super::checked_add_u64;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

fn for_each_valid_idx(validity: &Mask, len: usize, mut f: impl FnMut(usize)) {
    match validity.indices() {
        AllOr::All => {
            for idx in 0..len {
                f(idx);
            }
        }
        AllOr::None => {}
        AllOr::Some(indices) => {
            for &idx in indices {
                f(idx);
            }
        }
    }
}

fn accumulate_grouped_unsigned(partials: &mut [SumPartial], group_id: u32, value: u64) {
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Unsigned(acc)) => checked_add_u64(acc, value),
        Some(_) => vortex_panic!("unsigned sum state with non-unsigned input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_signed(partials: &mut [SumPartial], group_id: u32, value: i64) {
    let partial = &mut partials[group_id as usize];
    let saturated = match partial.current.as_mut() {
        None => return,
        Some(SumState::Signed(acc)) => checked_add_i64(acc, value),
        Some(_) => vortex_panic!("signed sum state with non-signed input"),
    };
    if saturated {
        partial.current = None;
    }
}

fn accumulate_grouped_float(partials: &mut [SumPartial], group_id: u32, value: f64) {
    if value.is_nan() {
        return;
    }

    match partials[group_id as usize].current.as_mut() {
        None => {}
        Some(SumState::Float(acc)) => *acc += value,
        Some(_) => vortex_panic!("float sum state with non-float input"),
    }
}

pub(super) fn accumulate_grouped_primitive(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = primitive
        .as_ref()
        .validity()?
        .execute_mask(primitive.as_ref().len(), ctx)?;
    match_each_native_ptype!(primitive.ptype(),
        unsigned: |T| {
            accumulate_grouped_primitive_unsigned::<T>(partials, primitive, group_ids, &validity);
            Ok(())
        },
        signed: |T| {
            accumulate_grouped_primitive_signed::<T>(partials, primitive, group_ids, &validity);
            Ok(())
        },
        floating: |T| {
            accumulate_grouped_primitive_float::<T>(partials, primitive, group_ids, &validity);
            Ok(())
        }
    )
}

fn accumulate_grouped_primitive_unsigned<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativePType + AsPrimitive<u64>,
{
    let values = primitive.as_slice::<T>();
    for_each_valid_idx(validity, values.len(), |idx| {
        accumulate_grouped_unsigned(partials, group_ids[idx], values[idx].as_());
    });
}

fn accumulate_grouped_primitive_signed<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativePType + AsPrimitive<i64>,
{
    let values = primitive.as_slice::<T>();
    for_each_valid_idx(validity, values.len(), |idx| {
        accumulate_grouped_signed(partials, group_ids[idx], values[idx].as_());
    });
}

fn accumulate_grouped_primitive_float<T>(
    partials: &mut [SumPartial],
    primitive: &PrimitiveArray,
    group_ids: &[u32],
    validity: &Mask,
) where
    T: NativePType + ToPrimitive,
{
    let values = primitive.as_slice::<T>();
    for_each_valid_idx(validity, values.len(), |idx| {
        let value = values[idx].to_f64().vortex_expect("float to f64");
        accumulate_grouped_float(partials, group_ids[idx], value);
    });
}

pub(super) fn accumulate_grouped_bool(
    partials: &mut [SumPartial],
    bools: &BoolArray,
    group_ids: &[u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let validity = bools
        .as_ref()
        .validity()?
        .execute_mask(bools.as_ref().len(), ctx)?;
    let values = bools.to_bit_buffer();
    for_each_valid_idx(&validity, values.len(), |idx| {
        if values.value(idx) {
            accumulate_grouped_unsigned(partials, group_ids[idx], 1);
        }
    });
    Ok(())
}
