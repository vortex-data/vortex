use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};

use crate::arrays::ConstantArray;
use crate::stats::{Precision, Stat};
use crate::{Array, ArrayExt, Encoding};

pub trait IsConstantFn<A> {
    fn is_constant(&self, array: A) -> VortexResult<bool>;
}

impl<E: Encoding> IsConstantFn<&dyn Array> for E
where
    E: for<'a> IsConstantFn<&'a E::Array>,
{
    fn is_constant(&self, array: &dyn Array) -> VortexResult<bool> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        IsConstantFn::is_constant(self, array_ref)
    }
}

pub fn is_constant(array: &dyn Array) -> VortexResult<bool> {
    if array.as_opt::<ConstantArray>().is_some() {
        return Ok(true);
    }

    if let Some(Precision::Exact(value)) = array.statistics().get_as::<bool>(Stat::IsConstant) {
        return Ok(value);
    }

    if array.len() <= 1 {
        return Ok(true);
    }

    if array.all_invalid()? {
        return Ok(true);
    }

    let max = array
        .statistics()
        .get_scalar(Stat::Max, array.dtype())
        .and_then(|p| p.some_exact());
    let min = array
        .statistics()
        .get_scalar(Stat::Min, array.dtype())
        .and_then(|p| p.some_exact());

    if let Some((min, max)) = min.zip(max) {
        if min == max && array.all_valid()? {
            return Ok(true);
        }
    }

    let is_constant = if let Some(is_constant_fn) = array.vtable().is_constant_fn() {
        is_constant_fn.is_constant(array)?
    } else {
        log::debug!(
            "No is_constant implementation found for {}",
            array.encoding()
        );

        let array = array.to_canonical()?;

        if let Some(is_constant_fn) = array.as_ref().vtable().is_constant_fn() {
            is_constant_fn.is_constant(array.as_ref())?
        } else {
            vortex_bail!(
                "No sum function for canonical array: {}",
                array.as_ref().encoding(),
            )
        }
    };

    array
        .statistics()
        .set_stat(Stat::IsConstant, Precision::Exact(is_constant.into()));

    Ok(is_constant)
}
