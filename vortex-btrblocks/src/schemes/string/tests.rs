// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::display::DisplayOptions;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| vortex_array::default_session_builder().build());

#[test]
fn test_strings() -> VortexResult<()> {
    let mut strings = Vec::new();
    for _ in 0..1024 {
        strings.push(Some("hello-world-1234"));
    }
    for _ in 0..1024 {
        strings.push(Some("hello-world-56789"));
    }
    let strings = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

    let array_ref = strings.into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 2048);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.dict(utf8, len=2048)");

    Ok(())
}

#[test]
fn test_sparse_nulls() -> VortexResult<()> {
    let mut strings = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::Nullable), 100);
    strings.append_nulls(99);

    strings.append_value("one little string");

    let strings = strings.finish_into_varbinview();

    let array_ref = strings.into_array();
    let btr = BtrBlocksCompressor::default();
    let compressed = btr.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert_eq!(compressed.len(), 100);

    let display = compressed
        .display_as(DisplayOptions::MetadataOnly)
        .to_string()
        .to_lowercase();
    assert_eq!(display, "vortex.sparse(utf8?, len=100)");

    Ok(())
}
