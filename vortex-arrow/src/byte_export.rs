// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The ready-made Arrow export plugin for `Utf8`/`Binary` encodings.

use std::sync::Arc;

use arrow_array::types::BinaryType;
use arrow_array::types::LargeBinaryType;
use arrow_array::types::LargeUtf8Type;
use arrow_array::types::Utf8Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;

use crate::ArrowExport;
use crate::ArrowExportKey;
use crate::ArrowExportOptions;
use crate::ArrowExportVTable;
use crate::ArrowExportVTableRef;
use crate::executor::byte::export_byte_array;

/// Exports an encoding's `Utf8`/`Binary` values through the offsets-based Arrow fast path.
///
/// The canonical conversion reaches Arrow's `Utf8`/`Binary` (and their `Large` counterparts) by
/// executing an array to a canonical `VarBinView` and then re-laying those views out into the
/// offsets and bytes buffers Arrow wants. An encoding that can write its values into a
/// `VarBinBuilder` itself — which is what
/// [`append_to_builder`](vortex_array::vtable::VTable::append_to_builder) does — pays for that
/// intermediate for nothing.
///
/// Registering this for such an encoding stops the export there and appends straight into the
/// builder for the target's offset width, skipping the `VarBinView` round trip. Dispatch only sees
/// the array passed to the Arrow session; a lazy operator above the encoding remains on the
/// canonical path unless it is executed before export.
///
/// The plugin claims only what it improves: an array whose dtype is `Utf8` or `Binary`, exported
/// to `Utf8`, `LargeUtf8`, `Binary`, or `LargeBinary`. Anything else — a view target, whose
/// canonical `VarBinView` layout is already what Arrow wants, or an export that named no type at
/// all — is left to the canonical conversion.
///
/// Register it for an encoding whose `append_to_builder` fills a `VarBinBuilder` directly. It is
/// the wrong choice for an encoding worth executing *through*: `Dict`, for instance, gathers
/// canonical values into the builder, and executing through it instead lets kernels such as
/// FSST's `Dict` parent kernel decode through the dictionary.
#[derive(Debug)]
pub struct ByteArrayExporter {
    encoding_id: ArrayId,
}

impl ByteArrayExporter {
    /// A plugin exporting the `Utf8`/`Binary` arrays of `encoding_id`.
    ///
    /// Register it with
    /// [`ArrowSession::register_exporter`](crate::ArrowSession::register_exporter).
    pub fn for_encoding(encoding_id: ArrayId) -> ArrowExportVTableRef {
        Arc::new(Self { encoding_id })
    }
}

impl ArrowExportVTable for ByteArrayExporter {
    fn export_key(&self) -> ArrowExportKey {
        ArrowExportKey::encoding(self.encoding_id)
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: Option<&Field>,
        _options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let Some(target) =
            target.filter(|_| matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_)))
        else {
            return Ok(ArrowExport::Unsupported(array));
        };

        match target.data_type() {
            DataType::Utf8 => export_byte_array::<Utf8Type>(array, ctx),
            DataType::LargeUtf8 => export_byte_array::<LargeUtf8Type>(array, ctx),
            DataType::Binary => export_byte_array::<BinaryType>(array, ctx),
            DataType::LargeBinary => export_byte_array::<LargeBinaryType>(array, ctx),
            _ => return Ok(ArrowExport::Unsupported(array)),
        }
        .map(ArrowExport::Exported)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::cast::AsArray;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_error::VortexResult;
    use vortex_fsst::FSST;
    use vortex_fsst::fsst_compress;
    use vortex_fsst::fsst_train_compressor;
    use vortex_onpair::DEFAULT_CONFIG;
    use vortex_onpair::OnPair;
    use vortex_onpair::onpair_compress;
    use vortex_session::VortexSession;
    use vortex_zstd::Zstd;

    use super::*;
    use crate::ArrowSessionExt;

    /// The values every test compresses, long enough that each encoding has something to work
    /// with.
    fn values() -> Vec<String> {
        (0..512)
            .map(|i| format!("value-{}-{}", i % 7, "x".repeat(i % 13)))
            .collect()
    }

    fn source() -> ArrayRef {
        let strings = values();
        VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array()
    }

    /// Compresses `source` with each encoding this crate can build in tests.
    #[derive(Clone, Copy, Debug)]
    enum Encoding {
        Fsst,
        OnPair,
        Zstd,
    }

    impl Encoding {
        fn encoding_id(self) -> ArrayId {
            match self {
                Encoding::Fsst => FSST.id(),
                Encoding::OnPair => OnPair.id(),
                Encoding::Zstd => Zstd.id(),
            }
        }

        fn compress(self, session: &VortexSession) -> VortexResult<ArrayRef> {
            let mut ctx = session.create_execution_ctx();
            let source = source();
            Ok(match self {
                Encoding::Fsst => {
                    let compressor = fsst_train_compressor(&source, &mut ctx)?;
                    fsst_compress(&source, &compressor, &mut ctx)?.into_array()
                }
                Encoding::OnPair => onpair_compress(&source, DEFAULT_CONFIG, &mut ctx)?,
                Encoding::Zstd => {
                    let view = source.execute::<VarBinViewArray>(&mut ctx)?;
                    Zstd::from_var_bin_view_without_dict(&view, 3, 8_192, &mut ctx)?.into_array()
                }
            })
        }
    }

    /// A session with the encodings registered but no exporter, so exports take the canonical
    /// path.
    fn session() -> VortexSession {
        let session = array_session();
        crate::initialize(&session);
        let arrays = session.arrays();
        arrays.register(FSST);
        arrays.register(OnPair);
        arrays.register(Zstd);
        drop(arrays);
        session
    }

    fn export(
        session: &VortexSession,
        array: ArrayRef,
        data_type: DataType,
    ) -> VortexResult<ArrowArrayRef> {
        let mut ctx = session.create_execution_ctx();
        let field = Field::new("s", data_type, array.dtype().is_nullable());
        session.arrow().execute_arrow(array, Some(&field), &mut ctx)
    }

    /// The exporter is a shortcut, not a different answer: every encoding it claims must export
    /// exactly what the canonical conversion would have produced.
    #[rstest]
    fn exports_match_the_canonical_conversion(
        #[values(Encoding::Fsst, Encoding::OnPair, Encoding::Zstd)] encoding: Encoding,
        #[values(
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary
        )]
        data_type: DataType,
    ) -> VortexResult<()> {
        let canonical_session = session();
        let expected = export(
            &canonical_session,
            encoding.compress(&canonical_session)?,
            data_type.clone(),
        )?;

        let plugin_session = session();
        plugin_session
            .arrow()
            .register_exporter(ByteArrayExporter::for_encoding(encoding.encoding_id()));
        let actual = export(
            &plugin_session,
            encoding.compress(&plugin_session)?,
            data_type.clone(),
        )?;

        assert_eq!(actual.data_type(), &data_type);
        assert_eq!(actual.to_data(), expected.to_data());
        Ok(())
    }

    /// The exporter claims the encoding itself. An array behind a lazy operator is not at that
    /// encoding yet, so it stays on the canonical path, which pushes the operator down first.
    #[test]
    fn values_survive_an_export_of_a_filtered_array() -> VortexResult<()> {
        use vortex_mask::Mask;

        let session = session();
        session
            .arrow()
            .register_exporter(ByteArrayExporter::for_encoding(FSST.id()));

        let strings = values();
        let selected = Mask::from_iter((0..strings.len()).map(|i| i % 3 == 0));
        let filtered = Encoding::Fsst
            .compress(&session)?
            .filter(selected)?
            .into_array();

        let arrow = export(&session, filtered, DataType::Utf8)?;

        let expected: Vec<&str> = strings
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(_, value)| value.as_str())
            .collect();
        let actual = arrow.as_string::<i32>();
        assert_eq!(
            (0..actual.len())
                .map(|i| actual.value(i))
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    /// An Arrow type the offsets-based fast path does not serve is left to the canonical
    /// conversion.
    #[rstest]
    #[case::view(Some(DataType::Utf8View))]
    #[case::no_requested_type(None)]
    fn other_targets_fall_through(#[case] data_type: Option<DataType>) -> VortexResult<()> {
        let session = session();
        session
            .arrow()
            .register_exporter(ByteArrayExporter::for_encoding(FSST.id()));

        let array = Encoding::Fsst.compress(&session)?;
        let mut ctx = session.create_execution_ctx();
        let field = data_type.map(|data_type| Field::new("s", data_type, false));
        let arrow = session
            .arrow()
            .execute_arrow(array, field.as_ref(), &mut ctx)?;

        assert_eq!(arrow.data_type(), &DataType::Utf8View);
        let actual = arrow.as_string_view();
        assert_eq!(
            (0..actual.len())
                .map(|i| actual.value(i))
                .collect::<Vec<_>>(),
            values()
        );
        Ok(())
    }
}
