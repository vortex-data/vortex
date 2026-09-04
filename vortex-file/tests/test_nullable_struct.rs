// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();

    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);

    session
});

#[tokio::test]
async fn nullable_struct_child_inherits_parent_validity() -> VortexResult<()> {
    // The second child value is deliberately valid and would match `a > 1`, but its parent struct
    // row is null. This is the representation produced by older writers and is valid Arrow data.
    let nullable_struct = StructArray::try_new(
        FieldNames::from(["a"]),
        vec![buffer![1i32, 2].into_array()],
        2,
        Validity::Array(BoolArray::from_iter([true, false]).into_array()),
    )?;
    let data = StructArray::try_new(
        FieldNames::from(["s"]),
        vec![nullable_struct.into_array()],
        2,
        Validity::NonNullable,
    )?
    .into_array();

    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .write(&mut bytes, data.to_array_stream())
        .await?;
    let file = SESSION
        .open_options()
        .open_buffer(ByteBuffer::from(bytes))?;

    let field = get_item("a", get_item("s", root()));
    let projected = file
        .scan()?
        .with_projection(field.clone())
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(
        projected.invalid_count(&mut SESSION.create_execution_ctx())?,
        1
    );

    let nulls = file
        .scan()?
        .with_filter(is_null(field.clone()))
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(nulls.len(), 1);

    let non_nulls = file
        .scan()?
        .with_filter(is_not_null(field.clone()))
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(non_nulls.len(), 1);

    let raw_child_match = file
        .scan()?
        .with_filter(gt(field, lit(1i32)))
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(raw_child_match.len(), 0);

    Ok(())
}
