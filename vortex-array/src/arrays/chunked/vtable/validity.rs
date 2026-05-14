// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::validity::Validity;

impl ValidityVTable<Chunked> for Chunked {
    fn validity(array: ArrayView<'_, Chunked>) -> VortexResult<Validity> {
        let validities = array
            .chunks()
            .iter()
            .map(|chunk| chunk.validity().map(|v| (v, chunk.len())))
            .try_collect()?;
        let Some(validity) = Validity::concat(validities) else {
            // If there are no chunks:
            return Ok(array.dtype().nullability().into());
        };

        Ok(validity)
    }
}
