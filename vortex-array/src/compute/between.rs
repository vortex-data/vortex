use vortex_error::{VortexError, VortexResult};

use crate::{Array, Encoding, IntoArray, IntoCanonical};

pub trait BetweenFn<A> {
    fn between(&self, arr: &A, lower: &Array, upper: &Array) -> VortexResult<Option<Array>>;
}

impl<E: Encoding> BetweenFn<Array> for E
where
    E: BetweenFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn between(&self, arr: &Array, lower: &Array, upper: &Array) -> VortexResult<Option<Array>> {
        let (arr_ref, encoding) = arr.try_downcast_ref::<E>()?;
        BetweenFn::between(encoding, arr_ref, lower, upper)
    }
}

pub fn between(
    arr: impl AsRef<Array>,
    lower: impl AsRef<Array>,
    upper: impl AsRef<Array>,
) -> VortexResult<Array> {
    let arr = arr.as_ref();
    let lower = lower.as_ref();
    let upper = upper.as_ref();

    if let Some(result) = arr
        .vtable()
        .between_fn()
        .and_then(|f| f.between(arr, lower, upper).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    let arr = arr.clone().into_canonical()?.into_array();

    if let Some(result) = arr
        .vtable()
        .between_fn()
        .and_then(|f| f.between(&arr, lower, upper).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    todo!("between")
}
