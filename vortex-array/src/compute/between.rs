use vortex_error::{VortexError, VortexResult};

use crate::compute::{binary_boolean, compare, BinaryOperator, Operator};
use crate::{Array, Encoding};

pub trait BetweenFn<A> {
    fn between(
        &self,
        arr: &A,
        lower: &Array,
        lower_op: Operator,
        upper: &Array,
        upper_op: Operator,
    ) -> VortexResult<Option<Array>>;
}

impl<E: Encoding> BetweenFn<Array> for E
where
    E: BetweenFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn between(
        &self,
        arr: &Array,
        lower: &Array,
        lower_op: Operator,
        upper: &Array,
        upper_op: Operator,
    ) -> VortexResult<Option<Array>> {
        let (arr_ref, encoding) = arr.try_downcast_ref::<E>()?;
        BetweenFn::between(encoding, arr_ref, lower, lower_op, upper, upper_op)
    }
}

pub fn between(
    arr: impl AsRef<Array>,
    lower: impl AsRef<Array>,
    lower_op: Operator,
    upper: impl AsRef<Array>,
    upper_op: Operator,
) -> VortexResult<Array> {
    let arr = arr.as_ref();
    let lower = lower.as_ref();
    let upper = upper.as_ref();

    if let Some(result) = arr
        .vtable()
        .between_fn()
        .and_then(|f| f.between(arr, lower, lower_op, upper, upper_op).transpose())
        .transpose()?
    {
        return Ok(result);
    }

    binary_boolean(
        &compare(lower, arr, Operator::Gt)?,
        &compare(arr, upper, Operator::Lt)?,
        BinaryOperator::And,
    )

    // println!("between {:?}", arr.encoding());
    // let arr = arr.clone().into_canonical()?.into_array();
    //
    // if let Some(result) = arr
    //     .vtable()
    //     .between_fn()
    //     .and_then(|f| f.between(&arr, lower, upper).transpose())
    //     .transpose()?
    // {
    //     return Ok(result);
    // }

    // todo!("between {:?}", arr.encoding())
}
