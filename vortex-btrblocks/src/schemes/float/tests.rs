// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::display::DisplayOptions;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer_mut;
use vortex_compressor::CascadingCompressor;
use vortex_error::VortexResult;
use vortex_fastlanes::RLE;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
use crate::schemes::float::FloatRLEScheme;
static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_empty() -> VortexResult<()> {
    let btr = BtrBlocksCompressor::default();
    let array = PrimitiveArray::new(Buffer::<f32>::empty(), Validity::NonNullable).into_array();
    let result = btr.compress(&array, &mut SESSION.create_execution_ctx())?;

    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_compress() -> VortexResult<()> {
    let mut values = buffer_mut![1.0f32; 1024];
    for i in 0..1024 {
        values[i] = (i % 50) as f32;
    }

    let array = values.into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 1024);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.dict(f32, len=1024)");

    Ok(())
}

#[test]
fn test_rle_compression() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut values = Vec::new();
    values.extend(iter::repeat_n(1.5f32, 100));
    values.extend(iter::repeat_n(2.7f32, 200));
    values.extend(iter::repeat_n(3.15f32, 150));

    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);

    let compressor = CascadingCompressor::new(vec![&FloatRLEScheme]);
    let compressed =
        compressor.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<RLE>());

    let expected = Buffer::copy_from(&values).into_array();
    assert_arrays_eq!(compressed, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_sparse_compression() -> VortexResult<()> {
    let mut array = PrimitiveBuilder::<f32>::with_capacity(Nullability::Nullable, 100);
    array.append_value(f32::NAN);
    array.append_value(-f32::NAN);
    array.append_value(f32::INFINITY);
    array.append_value(-f32::INFINITY);
    array.append_value(0.0f32);
    array.append_value(-0.0f32);
    array.append_nulls(90);

    let array = array.finish_into_primitive().into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 96);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.sparse(f32?, len=96)");

    Ok(())
}
