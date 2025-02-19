use itertools::Itertools;
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{invert, InvertFn};
use crate::{Array, IntoArray};

impl InvertFn<ChunkedArray> for ChunkedEncoding {
    fn invert(&self, array: &ChunkedArray) -> VortexResult<Array> {
        let chunks = array.chunks().map(|c| invert(&c)).try_collect()?;
        Ok(ChunkedArray::try_new_unchecked(chunks, array.dtype().clone()).into_array())
    }
}
