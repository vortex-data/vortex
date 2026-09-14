// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Int32Array;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use arrow_array::cast::AsArray;
use arrow_array::new_null_array;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use rstest::rstest;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::*;

const MARKER_INT: i32 = 7;
const MARKER_STR: &str = "marker";

/// An extension with no Arrow extension exporter, used to ensure an encoding exporter can
/// choose a physical Arrow type without schema inference succeeding first.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct UnmappedExtension;

impl ExtVTable for UnmappedExtension {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = i32;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("vortex.arrow.test.unmapped")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        _storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(0)
    }
}

/// A run of marker values, which the canonical conversion could never produce from the arrays
/// these tests build. That is what tells a test which of the two paths ran.
fn marker(data_type: &DataType, len: usize) -> VortexResult<ArrowArrayRef> {
    Ok(match data_type {
        DataType::Int32 => Arc::new(Int32Array::from(vec![MARKER_INT; len])),
        DataType::Utf8 => Arc::new(StringArray::from(vec![MARKER_STR; len])),
        DataType::Utf8View => Arc::new(StringViewArray::from(vec![MARKER_STR; len])),
        data_type => vortex_bail!("no marker array for {data_type}"),
    })
}

/// Exports every array of one encoding as [`marker`] values.
#[derive(Debug)]
struct MarkerExporter {
    key: ArrowExportKey,
    /// The Arrow type it produces when an export requests none.
    data_type: DataType,
}

impl MarkerExporter {
    /// Claims `encoding` for exports to `data_type` only.
    fn to_data_type(encoding: ArrayId, data_type: DataType) -> ArrowExportVTableRef {
        Arc::new(Self {
            key: ArrowExportKey::encoding_to(encoding, data_type.clone()),
            data_type,
        })
    }

    /// Claims `encoding` for every export, producing `data_type` when none is requested.
    fn for_encoding(encoding: ArrayId, data_type: DataType) -> ArrowExportVTableRef {
        Arc::new(Self {
            key: ArrowExportKey::encoding(encoding),
            data_type,
        })
    }
}

impl ArrowExportVTable for MarkerExporter {
    fn export_key(&self) -> ArrowExportKey {
        self.key.clone()
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        _options: &ArrowExportOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let data_type = target.map_or(&self.data_type, |target| target.data_type());
        marker(data_type, array.len()).map(ArrowExport::Exported)
    }
}

/// Counts how often it is probed and always defers, so a test can assert the plugins ahead of
/// a claiming one still got their turn.
#[derive(Debug)]
struct DeferringExporter {
    key: ArrowExportKey,
    probes: Arc<AtomicUsize>,
}

impl DeferringExporter {
    fn registered(key: ArrowExportKey) -> (ArrowExportVTableRef, Arc<AtomicUsize>) {
        let probes = Arc::new(AtomicUsize::new(0));
        let exporter = Arc::new(Self {
            key,
            probes: Arc::clone(&probes),
        });
        (exporter, probes)
    }
}

impl ArrowExportVTable for DeferringExporter {
    fn export_key(&self) -> ArrowExportKey {
        self.key.clone()
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: Option<&Field>,
        _options: &ArrowExportOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        self.probes.fetch_add(1, Ordering::Relaxed);
        Ok(ArrowExport::Unsupported(array))
    }
}

/// Exports an Arrow type or a length other than the one asked for, to exercise the checks on
/// what a plugin hands back.
#[derive(Debug)]
struct MisbehavingExporter {
    key: ArrowExportKey,
    exported_data_type: DataType,
    len: usize,
}

impl ArrowExportVTable for MisbehavingExporter {
    fn export_key(&self) -> ArrowExportKey {
        self.key.clone()
    }

    fn execute_arrow(
        &self,
        _array: ArrayRef,
        _target: Option<&Field>,
        _options: &ArrowExportOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        Ok(ArrowExport::Exported(new_null_array(
            &self.exported_data_type,
            self.len,
        )))
    }
}

fn session() -> VortexSession {
    let session = array_session();
    // Materialize the lazily-created Arrow session up front so registrations and exports share
    // one instance.
    crate::initialize(&session);
    session
}

/// `["a", "b", "a"]`, dictionary encoded.
fn utf8_dict() -> VortexResult<ArrayRef> {
    Ok(DictArray::try_new(
        PrimitiveArray::from_iter([0u8, 1, 0]).into_array(),
        VarBinViewArray::from_iter_str(["a", "b"]).into_array(),
    )?
    .into_array())
}

/// `[10, 20, 10]`, dictionary encoded.
fn primitive_dict() -> VortexResult<ArrayRef> {
    Ok(DictArray::try_new(
        PrimitiveArray::from_iter([0u8, 1, 0]).into_array(),
        PrimitiveArray::from_iter([10i32, 20]).into_array(),
    )?
    .into_array())
}

fn export(
    session: &VortexSession,
    array: ArrayRef,
    target: Option<&Field>,
) -> VortexResult<ArrowArrayRef> {
    let mut ctx = session.create_execution_ctx();
    session.arrow().execute_arrow(array, target, &mut ctx)
}

/// A plugin registered for an encoding claims its arrays whatever their dtype: the same plugin
/// exports a `Utf8` and a `Primitive` dictionary, neither of which is executed to a canonical
/// encoding first.
#[rstest]
#[case::utf8(utf8_dict(), DataType::Utf8)]
#[case::primitive(primitive_dict(), DataType::Int32)]
fn encoding_exporter_claims_its_encoding(
    #[case] array: VortexResult<ArrayRef>,
    #[case] data_type: DataType,
) -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), data_type.clone()));

    let field = Field::new("dict", data_type.clone(), false);
    let arrow = export(&session, array?, Some(&field))?;

    assert_eq!(arrow.data_type(), &data_type);
    assert_eq!(arrow.len(), 3);
    assert_marker(&arrow);
    Ok(())
}

/// The Arrow type is half the key, so a plugin that named another one is never consulted and
/// the canonical conversion exports the dictionary's real values.
#[test]
fn another_arrow_type_is_a_different_key() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Int32));

    let field = Field::new("dict", DataType::Utf8, false);
    let arrow = export(&session, utf8_dict()?, Some(&field))?;

    assert_eq!(
        string_values(&arrow),
        ["a", "b", "a"],
        "expected the canonical conversion to export the dictionary"
    );
    Ok(())
}

/// A plugin that named no Arrow type claims every export of its encoding, and picks the type
/// itself when none was requested.
#[rstest]
#[case::requested(Some(DataType::Utf8), DataType::Utf8)]
#[case::chosen_by_the_plugin(None, DataType::Utf8View)]
fn encoding_exporter_without_a_data_type_claims_every_export(
    #[case] requested: Option<DataType>,
    #[case] expected: DataType,
) -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::for_encoding(Dict.id(), DataType::Utf8View));

    let field = requested.map(|data_type| Field::new("dict", data_type, false));
    let arrow = export(&session, utf8_dict()?, field.as_ref())?;

    assert_eq!(arrow.data_type(), &expected);
    assert_marker(&arrow);
    Ok(())
}

/// An export that requests no Arrow type reaches only the plugins that named none, so one
/// registered for a specific type stays out of the way.
#[test]
fn an_export_of_no_particular_type_skips_type_specific_plugins() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8View));

    let arrow = export(&session, utf8_dict()?, None)?;

    assert_eq!(arrow.data_type(), &DataType::Utf8View);
    let strings = arrow.as_string_view();
    assert_eq!(
        (0..3).map(|i| strings.value(i)).collect::<Vec<_>>(),
        ["a", "b", "a"],
        "expected the canonical conversion to export the dictionary"
    );
    Ok(())
}

/// A plugin that chooses its own Arrow type does not need schema inference to understand the
/// dtype. This lets an encoding export an extension dtype for which no extension plugin exists.
#[test]
fn encoding_exporter_can_handle_a_dtype_schema_inference_cannot() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::for_encoding(
            Extension.id(),
            DataType::Int32,
        ));

    let storage = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = ExtensionArray::try_new_from_vtable(UnmappedExtension, EmptyMetadata, storage)?
        .into_array();
    let arrow = export(&session, array, None)?;

    assert_eq!(arrow.data_type(), &DataType::Int32);
    assert_marker(&arrow);
    Ok(())
}

/// Eager initialization must not replace an Arrow session that encoding initializers already
/// materialized and populated.
#[test]
fn initialize_preserves_lazily_registered_exporters() -> VortexResult<()> {
    let session = array_session();
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

    crate::initialize(&session);

    let field = Field::new("dict", DataType::Utf8, false);
    let arrow = export(&session, utf8_dict()?, Some(&field))?;
    assert_marker(&arrow);
    Ok(())
}

/// The plugins that named the Arrow type being exported to are more specific, so they are
/// tried before those that claim the encoding whatever the type.
#[test]
fn type_specific_plugins_are_tried_first() -> VortexResult<()> {
    let session = session();
    let (deferring, probes) =
        DeferringExporter::registered(ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8));
    session.arrow().register_exporter(deferring);
    session
        .arrow()
        .register_exporter(MarkerExporter::for_encoding(Dict.id(), DataType::Utf8));

    let field = Field::new("dict", DataType::Utf8, false);
    let arrow = export(&session, utf8_dict()?, Some(&field))?;

    assert_eq!(probes.load(Ordering::Relaxed), 1);
    assert_marker(&arrow);
    Ok(())
}

/// A plugin that defers hands the array on, and with no other plugin to claim it the canonical
/// conversion exports the dictionary's real values.
#[test]
fn deferring_encoding_exporter_falls_through() -> VortexResult<()> {
    let session = session();
    let (deferring, probes) =
        DeferringExporter::registered(ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8));
    session.arrow().register_exporter(deferring);

    let field = Field::new("dict", DataType::Utf8, false);
    let arrow = export(&session, utf8_dict()?, Some(&field))?;

    assert_eq!(probes.load(Ordering::Relaxed), 1);
    assert_eq!(string_values(&arrow), ["a", "b", "a"]);
    Ok(())
}

/// Plugins registered under one key are tried in registration order.
#[test]
fn encoding_exporters_are_tried_in_registration_order() -> VortexResult<()> {
    let session = session();
    let (deferring, probes) =
        DeferringExporter::registered(ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8));
    session.arrow().register_exporter(deferring);
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

    let field = Field::new("dict", DataType::Utf8, false);
    let arrow = export(&session, utf8_dict()?, Some(&field))?;

    assert_eq!(probes.load(Ordering::Relaxed), 1);
    assert_marker(&arrow);
    Ok(())
}

/// Columns nested inside a struct route back through the session, so their encodings reach
/// their plugins too.
#[test]
fn encoding_exporter_claims_a_nested_column() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

    let array = StructArray::try_new(
        FieldNames::from(["dict"]),
        vec![utf8_dict()?],
        3,
        Validity::NonNullable,
    )?
    .into_array();

    let field = Field::new(
        "row",
        DataType::Struct(Fields::from(vec![Field::new(
            "dict",
            DataType::Utf8,
            false,
        )])),
        false,
    );
    let arrow = export(&session, array, Some(&field))?;

    assert_marker(arrow.as_struct().column(0));
    Ok(())
}

/// An encoding with no registered plugin is exported by the canonical conversion, as before.
#[test]
fn unregistered_encodings_use_the_canonical_conversion() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(MarkerExporter::to_data_type(Dict.id(), DataType::Utf8));

    let field = Field::new("s", DataType::Utf8, false);
    let arrow = export(
        &session,
        VarBinViewArray::from_iter_str(["a", "b", "a"]).into_array(),
        Some(&field),
    )?;

    assert_eq!(string_values(&arrow), ["a", "b", "a"]);
    Ok(())
}

/// A plugin that drops or invents rows is rejected rather than silently corrupting the export.
#[test]
fn encoding_exporter_must_preserve_length() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(Arc::new(MisbehavingExporter {
            key: ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8),
            exported_data_type: DataType::Utf8,
            len: 2,
        }));

    let field = Field::new("dict", DataType::Utf8, true);
    let err = export(&session, utf8_dict()?, Some(&field))
        .expect_err("expected a length mismatch to be rejected");

    assert!(
        err.to_string().contains("length does not match"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// A plugin that exports an Arrow type other than the one requested is rejected too: its
/// caller asked for a specific type and would otherwise get another.
#[test]
fn encoding_exporter_must_honor_the_requested_type() -> VortexResult<()> {
    let session = session();
    session
        .arrow()
        .register_exporter(Arc::new(MisbehavingExporter {
            key: ArrowExportKey::encoding_to(Dict.id(), DataType::Utf8),
            exported_data_type: DataType::Int32,
            len: 3,
        }));

    let field = Field::new("dict", DataType::Utf8, true);
    let err = export(&session, utf8_dict()?, Some(&field))
        .expect_err("expected the wrong Arrow type to be rejected");

    assert!(
        err.to_string().contains("but Utf8 was requested"),
        "unexpected error: {err}"
    );
    Ok(())
}

fn string_values(arrow: &ArrowArrayRef) -> Vec<&str> {
    let strings = arrow.as_string::<i32>();
    (0..strings.len()).map(|i| strings.value(i)).collect()
}

fn assert_marker(arrow: &ArrowArrayRef) {
    match arrow.data_type() {
        DataType::Utf8 => {
            assert!(
                string_values(arrow)
                    .iter()
                    .all(|value| *value == MARKER_STR),
                "expected the exporter's marker values, got {arrow:?}"
            );
        }
        DataType::Utf8View => {
            let strings = arrow.as_string_view();
            assert!(
                (0..strings.len()).all(|i| strings.value(i) == MARKER_STR),
                "expected the exporter's marker values, got {arrow:?}"
            );
        }
        DataType::Int32 => {
            let ints = arrow.as_primitive::<Int32Type>();
            assert!(
                ints.values().iter().all(|value| *value == MARKER_INT),
                "expected the exporter's marker values, got {arrow:?}"
            );
        }
        data_type => panic!("unexpected marker type {data_type}"),
    }
}
