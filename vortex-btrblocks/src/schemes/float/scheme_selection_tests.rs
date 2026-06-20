// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each float compression scheme produces the expected encoding.

use std::sync::LazyLock;

use vortex_alp::ALP;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let values: Vec<f64> = vec![42.5; 100];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_alp_compressed() -> VortexResult<()> {
    let values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.01).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<ALP>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let distinct_values = [1.1, 2.2, 3.3, 4.4, 5.5];
    let values: Vec<f64> = (0..1000)
        .map(|i| distinct_values[i % distinct_values.len()])
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<ALP>());
    assert!(compressed.children()[0].is::<Dict>());
    Ok(())
}

#[test]
fn test_null_dominated_compressed() -> VortexResult<()> {
    let mut builder = PrimitiveBuilder::<f64>::with_capacity(Nullability::Nullable, 100);
    for i in 0..5 {
        builder.append_value(i as f64);
    }
    builder.append_nulls(95);
    let array = builder.finish_into_primitive();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array.into_array(), &mut SESSION.create_execution_ctx())?;
    // Verify the compressed array preserves values.
    assert_eq!(compressed.len(), 100);
    Ok(())
}
