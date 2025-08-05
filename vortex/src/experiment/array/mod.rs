// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// mod bool;
mod primitive;

use crate::experiment::array::primitive::export_primitive;
use crate::experiment::buffers::BufferId;
use crate::experiment::encodings::Encoding;
use bitvec::order::Msb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use vortex_array::Canonical;
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

pub struct Array {
    len: usize,
    dtype: DType,
    stats_set: StatsSet,
    encoding: Box<dyn Encoding>,
}

impl Array {
    pub fn new(len: usize, dtype: DType, stats_set: StatsSet, encoding: Box<dyn Encoding>) -> Self {
        Array {
            len,
            dtype,
            stats_set,
            encoding,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn to_canonical(&self, mask: &Mask) -> VortexResult<Canonical> {
        match &self.dtype {
            // DType::Bool(_) => export_bool(self).map(Canonical::Bool),
            DType::Primitive(..) => export_primitive(self, mask).map(Canonical::Primitive),
            _ => vortex_bail!("Unsupported dtype for canonical conversion: {}", self.dtype),
        }
    }
}
