use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::compute::{MinMaxKernel, MinMaxKernelAdapter, MinMaxResult, min_max};
use crate::partial_ord::{partial_max, partial_min};
use crate::register_kernel;

impl MinMaxKernel for ChunkedEncoding {
    fn min_max(&self, array: &ChunkedArray) -> VortexResult<Option<MinMaxResult>> {
        let mut min_max_all_null = true;
        let res = array
            .array_iterator()
            .map(|chunk| {
                let chunk = chunk?;
                if let Some(min_max) = min_max(&chunk)? {
                    min_max_all_null = false;
                    Ok((Some(min_max.min), Some(min_max.max)))
                } else {
                    Ok((None, None))
                }
            })
            .collect::<VortexResult<Vec<_>>>()?;

        // There are no chunks that have min/max stats, so return early
        if min_max_all_null {
            return Ok(None);
        }

        let (min_values, max_values): (Vec<Option<Scalar>>, Vec<Option<Scalar>>) =
            res.into_iter().unzip();

        Ok(Some(MinMaxResult {
            min: min_values
                .into_iter()
                .flatten()
                // This is None iff all the values `None` (refuted above) or partial_min returns None
                .fold(None, |acc, x| {
                    if let Some(acc) = acc {
                        partial_min(x, acc)
                    } else {
                        Some(x)
                    }
                })
                .ok_or_else(|| {
                    vortex_err!("Incomparable scalars (from partial_min), this is likely a bug",)
                })?,
            max: max_values
                .into_iter()
                .flatten()
                .fold(None, |acc, x| {
                    if let Some(acc) = acc {
                        partial_max(x, acc)
                    } else {
                        Some(x)
                    }
                })
                .ok_or_else(|| vortex_err!("Incomparable scalars, this is likely a bug"))?,
        }))
    }
}

register_kernel!(MinMaxKernelAdapter(ChunkedEncoding).lift());
