use itertools::Itertools;
use vortex_error::VortexResult;

use crate::array::{ChunkedArray, ChunkedEncoding};
use crate::compute::{invert, InvertFn};
use crate::{ArrayData, IntoArrayData};

impl InvertFn<ChunkedArray> for ChunkedEncoding {
    fn invert(&self, array: &ChunkedArray) -> VortexResult<ArrayData> {
        let chunks = array.chunks().map(|c| invert(&c)).try_collect()?;
        ChunkedArray::try_new(chunks, array.dtype().clone()).map(|a| a.into_array())
    }
}
