// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File round trips for the experimental extension layout.

use futures::TryStreamExt;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ForeignExtDType;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar::Scalar;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_edition::ComponentKind;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_edition::declarations::unstable::UNSTABLE;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::Expression;
use vortex_expr::fill_null;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::is_not_null;
use vortex_expr::lit;
use vortex_expr::pack;
use vortex_expr::root;
use vortex_functions::expr::ext_storage;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::extension::Extension;
use vortex_layout::layouts::struct_::Struct;
use vortex_layout::scan::v2;
use vortex_layout::session::LayoutSession;
use vortex_layout::session::LayoutSessionExt;
use vortex_scan::execution::driver::ScanDriver;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;

static POINT_ID: CachedId = CachedId::new("test.point");

fn session(unstable: bool, opted_in: bool) -> VortexResult<VortexSession> {
    let session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with_tokio();
    crate::register_default_encodings(&session);
    vortex_functions::initialize(&session);
    session.dtypes().register(ForeignExtDType::new(*POINT_ID));
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.editions().declare_inclusion(EditionInclusion::new(
        ComponentKind::DType,
        &*POINT_ID,
        CORE_2026_08_3,
    ))?;
    session.enable_edition(CORE_2026_08_3)?;
    if unstable {
        session.enable_edition(UNSTABLE)?;
    }
    if opted_in {
        session.layouts().decompose_extension(*POINT_ID);
    }

    Ok(session)
}

fn points(validity: Validity) -> VortexResult<ArrayRef> {
    let storage = StructArray::try_from_iter_with_validity(
        [
            ("x", buffer![1i32, 99, 3].into_array()),
            ("y", buffer![10i64, 990, 30].into_array()),
        ],
        validity,
    )?;
    let dtype = ForeignExtDType::from_parts(*POINT_ID, vec![1], storage.dtype().clone())?;
    Ok(ExtensionArray::try_new(dtype, storage.into_array())?.into_array())
}

async fn write_points(session: &VortexSession, points: ArrayRef) -> VortexResult<VortexFile> {
    let input = StructArray::try_from_iter([("point", points)])?.into_array();
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, input.to_array_stream())
        .await?;
    session.open_options().open_buffer(buffer)
}

fn point() -> Expression {
    get_item("point", root())
}

fn fallback_point() -> VortexResult<Scalar> {
    let storage_dtype = DType::Struct(
        StructFields::from_iter([
            ("x", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ("y", DType::Primitive(PType::I64, Nullability::NonNullable)),
        ]),
        Nullability::NonNullable,
    );
    let ext_dtype = ForeignExtDType::from_parts(*POINT_ID, vec![1], storage_dtype.clone())?;
    Ok(Scalar::extension_ref(
        ext_dtype,
        Scalar::struct_(storage_dtype, [Scalar::from(5i32), Scalar::from(50i64)]),
    ))
}

async fn assert_classic_and_v2_scan(
    session: &VortexSession,
    file: &VortexFile,
    projection: Expression,
    filter: Option<Expression>,
    expected: ArrayRef,
) -> VortexResult<()> {
    let projection = projection.bind(file.dtype())?;
    let filter = filter.map(|expr| expr.bind(file.dtype())).transpose()?;

    let classic = file
        .scan()?
        .with_projection(projection.clone())
        .with_some_filter(filter.clone())
        .into_array_stream()?
        .read_all()
        .await?;
    assert_arrays_eq!(classic, expected, &mut session.create_execution_ctx());

    let scan = v2::ScanBuilder::new(session.clone(), file.scan_file())
        .with_projection(projection)
        .with_some_filter(filter);
    let dtype = scan.dtype().clone();
    let chunks: Vec<ArrayRef> = ScanDriver::new(session.handle(), 1, true)
        .run(scan)
        .try_collect()
        .await?;
    let v2 = ChunkedArray::try_new(chunks, dtype)?.into_array();
    assert_arrays_eq!(v2, expected, &mut session.create_execution_ctx());

    Ok(())
}

#[rstest]
#[case::core_opted_in(false, true)]
#[case::unstable_default(true, false)]
#[case::unstable_opted_in(true, true)]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_extension_layout_policy(
    #[case] unstable: bool,
    #[case] opted_in: bool,
) -> VortexResult<()> {
    let session = session(unstable, opted_in)?;
    let points = points(Validity::NonNullable)?;
    let file = write_points(&session, points.clone()).await?;
    let layout = file
        .footer()
        .layout()
        .slot(1)?
        .ok_or_else(|| vortex_err!("missing point field"))?;

    assert_eq!(layout.is::<Extension>(), unstable && opted_in);
    if let Some(extension) = layout.as_opt::<Extension>() {
        assert!(extension.storage()?.is::<Struct>());
    }

    let actual = file
        .scan()?
        .with_projection(point().bind(file.dtype())?)
        .into_array_stream()?
        .read_all()
        .await?;
    assert_arrays_eq!(actual, points, &mut session.create_execution_ctx());

    Ok(())
}

#[rstest]
#[case::whole_value(point(), None, None)]
#[case::field(get_item("x", ext_storage(point())), None, Some(vec![Some(1), None, Some(3)]))]
#[case::storage_filter(
    get_item("x", ext_storage(point())),
    Some(gt(get_item("x", ext_storage(point())), lit(2i32))),
    Some(vec![Some(3)]),
)]
#[case::null_filter(
    get_item("x", ext_storage(point())),
    Some(is_not_null(point())),
    Some(vec![Some(1), Some(3)]),
)]
#[case::whole_value_filter(
    get_item("x", ext_storage(point())),
    Some(gt(
        get_item("x", ext_storage(fill_null(point(), lit(fallback_point()?)))),
        lit(2i32),
    )),
    Some(vec![None, Some(3)]),
)]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_nullable_extension_layout_scans(
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
    #[case] expected_field: Option<Vec<Option<i32>>>,
) -> VortexResult<()> {
    let session = session(true, true)?;
    let points = points(Validity::from_iter([true, false, true]))?;
    let file = write_points(&session, points.clone()).await?;
    let expected = expected_field.map_or(points, |field| {
        PrimitiveArray::from_option_iter(field).into_array()
    });

    assert_classic_and_v2_scan(&session, &file, projection, filter, expected).await
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn test_nullable_extension_layout_mixed_projection() -> VortexResult<()> {
    let session = session(true, true)?;
    let points = points(Validity::from_iter([true, false, true]))?;
    let file = write_points(&session, points.clone()).await?;
    let projection = pack(
        [
            ("point", point()),
            ("x", get_item("x", ext_storage(point()))),
        ],
        Nullability::NonNullable,
    );
    let expected = StructArray::try_from_iter([
        ("point", points),
        (
            "x",
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
        ),
    ])?
    .into_array();

    assert_classic_and_v2_scan(&session, &file, projection, None, expected).await
}
