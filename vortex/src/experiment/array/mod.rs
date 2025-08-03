// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod primitive;

use crate::experiment::array::bool::export_bool;
use crate::experiment::encodings::{BufferId, Encoding};
use vortex_array::Canonical;
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_utils::aliases::hash_map::HashMap;

pub struct Array {
    len: usize,
    dtype: DType,
    stats_set: StatsSet,
    encoding: Box<dyn Encoding>,

    buffers: HashMap<BufferId, ByteBuffer>,
}

impl Array {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn to_canonical(&self) -> VortexResult<Canonical> {
        match &self.dtype {
            DType::Bool(n) => export_bool(self).map(Canonical::Bool),
            _ => vortex_bail!("Unsupported dtype for canonical conversion: {}", self.dtype),
        }
    }
}
