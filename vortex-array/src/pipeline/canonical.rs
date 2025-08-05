// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::pipeline::ToPipeline;
use crate::pipeline::export::PrimitiveExporter;
use crate::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

pub fn export_primitive<P: ToPipeline>(pipeline: P) -> VortexResult<PrimitiveArray> {
    let len = pipeline.len();
    let mut elements = ByteBufferMut::with_capacity(len);
    PrimitiveExporter::new(&(), pipeline).export_all(|vec| {
        elements.extend_from_slice(vec.as_ref());
    });
    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}
