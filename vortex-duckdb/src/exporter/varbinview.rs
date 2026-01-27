// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::sync::Arc;

use itertools::Itertools;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::VarBinViewArrayParts;
use vortex::array::arrays::build_views::BinaryView;
use vortex::array::arrays::build_views::Inlined;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::LogicalType;
use crate::duckdb::Vector;
use crate::duckdb::VectorBuffer;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::validity;

struct VarBinViewExporter {
    views: Buffer<BinaryView>,
    buffers: Arc<[ByteBuffer]>,
    vector_buffers: Vec<VectorBuffer>,
}

pub(crate) fn new_exporter(
    array: VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let VarBinViewArrayParts {
        validity,
        dtype,
        views,
        buffers,
    } = array.into_parts();
    let validity = validity.to_array(len).execute::<Mask>(ctx)?;
    if validity.all_false() {
        return Ok(all_invalid::new_exporter(
            len,
            &LogicalType::try_from(dtype)?,
        ));
    }
    Ok(validity::new_exporter(
        validity,
        Box::new(VarBinViewExporter {
            views,
            buffers: buffers.clone(),
            vector_buffers: buffers.iter().cloned().map(VectorBuffer::new).collect_vec(),
        }),
    ))
}

impl ColumnExporter for VarBinViewExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Copy the views into place.
        for (mut_view, view) in unsafe { vector.as_slice_mut::<PtrBinaryView>(len) }
            .iter_mut()
            .zip(to_ptr_binary_view(
                self.views[offset..offset + len].iter(),
                &self.buffers,
            ))
        {
            *mut_view = view;
        }

        // We register our buffers zero-copy with DuckDB and re-use them in each vector.
        for buffer in &self.vector_buffers {
            vector.add_string_vector_buffer(buffer);
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
// See `BinaryView`
union PtrBinaryView {
    // Numeric representation. This is logically `u128`, but we split it into the high and low
    // bits to preserve the alignment.
    le_bytes: [u8; 16],

    // Inlined representation: strings <= 12 bytes
    inlined: Inlined,

    // Reference type: strings > 12 bytes.
    _ref: PtrRef,
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
struct PtrRef {
    size: u32,
    prefix: [u8; 4],
    ptr: *const c_char,
}

fn to_ptr_binary_view<'a>(
    view: impl Iterator<Item = &'a BinaryView>,
    buffers: &[ByteBuffer],
) -> impl Iterator<Item = PtrBinaryView> {
    view.map(|v| {
        if v.is_inlined() {
            PtrBinaryView {
                inlined: *v.as_inlined(),
            }
        } else {
            let view = v.as_view();
            PtrBinaryView {
                _ref: PtrRef {
                    size: v.len(),
                    prefix: view.prefix,
                    // TODO(joe) verify this.
                    ptr: unsafe {
                        buffers[view.buffer_index as usize]
                            .as_ptr()
                            .add(view.offset as usize)
                            .cast()
                    },
                },
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use Nullability::Nullable;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::error::VortexResult;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;

    use crate::LogicalType;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::exporter::varbinview::new_exporter;

    #[test]
    fn all_invalid_varbinview() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter([Option::<&str>::None; 4], DType::Utf8(Nullable));

        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        new_exporter(arr, &mut SESSION.create_execution_ctx())?.export(
            0,
            3,
            &mut chunk.get_vector(0),
        )?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT VARCHAR: 3 = [ NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn all_invalid_varbinview_section() -> VortexResult<()> {
        let arr =
            VarBinViewArray::from_iter([None, None, None, Some("Hey")], DType::Utf8(Nullable));

        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        new_exporter(arr, &mut SESSION.create_execution_ctx())?.export(
            0,
            3,
            &mut chunk.get_vector(0),
        )?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT VARCHAR: 3 = [ NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn partial_invalid_varbinview_section() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter(
            [None, None, Some("Hi"), Some("Hey")],
            DType::Utf8(Nullable),
        );

        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        new_exporter(arr, &mut SESSION.create_execution_ctx())?.export(
            0,
            3,
            &mut chunk.get_vector(0),
        )?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 3 = [ NULL, NULL, Hi]
"#
        );
        Ok(())
    }
}
