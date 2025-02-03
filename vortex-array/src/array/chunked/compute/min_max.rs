use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{min_max, MinMaxFn, MinMaxResult};
use crate::{partial_max, partial_min};

impl MinMaxFn<ChunkedArray> for ChunkedEncoding {
    fn min_max(&self, array: &ChunkedArray) -> VortexResult<MinMaxResult> {
        let res = array
            .array_iterator()
            .map(|chunk| {
                let chunk = chunk?;
                min_max(chunk)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (min_values, max_values): (Vec<Option<Scalar>>, Vec<Option<Scalar>>) =
            res.into_iter().unzip();

        Ok((
            min_values
                .into_iter()
                .flatten()
                .fold(None, |acc, x| acc.and_then(|acc| partial_min(x, acc))),
            max_values
                .into_iter()
                .flatten()
                .fold(None, |acc, x| acc.and_then(|acc| partial_max(x, acc))),
        ))
    }
}
