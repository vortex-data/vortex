// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::stats::ArrayStats;
use crate::vxo::{Array2, ArrayVTable, ArrayView, VTable};
use crate::EncodingId;
use vortex_buffer::Buffer;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType, PType};
use vortex_error::VortexResult;

/// A primitive array represents primitive data stored in a contiguous buffer.
struct Primitive;

impl Primitive {
    pub fn from_buffer<T: NativePType>(buffer: Buffer<T>) -> Array2 {
        let dtype = DType::Primitive(T::PTYPE, NonNullable);
        let len = buffer.len();
        let data = Data { ptype: T::PTYPE };
        // SAFETY: we ensure that the dtype, length, and data are consistent.
        unsafe {
            Array2::from_parts_unchecked(
                ArrayVTable::from_static(&Primitive),
                Box::new(data),
                dtype,
                len,
                vec![],
                ArrayStats::default(),
            )
        }
    }
}

struct Data {
    ptype: PType,
}

impl VTable for Primitive {
    type Instance = Data;

    fn id(&self) -> EncodingId {
        EncodingId::from("vortex.primitive")
    }

    fn validate(&self, expr: &ArrayView<Self>) -> VortexResult<()> {
        if expr.dtype().eq_ignore_nullability()
    }
}
