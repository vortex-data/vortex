// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrowSession`] — pluggable Vortex ↔ Arrow conversion session facet.

use std::any::Any;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parking_lot::RwLock;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::registry::Id;
use vortex_session::registry::Registry;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayId;
use crate::arrow::canonical::CanonicalArrowEncoder;
use crate::arrow::decoder::ArrowDecoderRef;
use crate::arrow::decoders::canonical::CanonicalArrowDTypeReader;
use crate::arrow::decoders::canonical::CanonicalArrowDecoder;
use crate::arrow::dtype_converter::ArrowDTypeConverterRef;
use crate::arrow::dtype_converter::ArrowDTypeReaderRef;
use crate::arrow::encoder::ArrowEncoderRef;
use crate::arrow::encoders::list::ListArrowEncoder;
use crate::arrow::encoders::temporal::TemporalArrowDTypeConverter;
use crate::arrow::encoders::temporal::TemporalArrowEncoder;
use crate::arrow::encoders::varbin::VarBinArrowEncoder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::Date;
use crate::extension::datetime::Time;
use crate::extension::datetime::Timestamp;

/// Registry for [`crate::arrow::ArrowEncoder`]s keyed by [`ArrayId`] (encoding-keyed dispatch).
pub type ArrowEncoderByEncodingRegistry = Registry<ArrowEncoderRef>;

/// Registry for [`crate::arrow::ArrowEncoder`]s keyed by [`ExtId`] (extension-keyed dispatch).
pub type ArrowEncoderByExtensionRegistry = Registry<ArrowEncoderRef>;

/// Registry for [`crate::arrow::ArrowDTypeConverter`]s keyed by [`ExtId`].
pub type ArrowDTypeConverterRegistry = Registry<ArrowDTypeConverterRef>;

/// Session facet for Vortex ↔ Arrow conversion plugins.
///
/// `ArrowSession` holds four kinds of plugin registries:
///
/// - **encoder_by_encoding** — [`crate::arrow::ArrowEncoder`]s keyed by [`ArrayId`]. Consulted
///   first during forward (Vortex → Arrow) array conversion, before canonicalization.
/// - **encoder_by_extension** — [`crate::arrow::ArrowEncoder`]s keyed by [`ExtId`]. Consulted
///   by the canonical encoder when handling [`crate::dtype::DType::Extension`] arrays.
/// - **dtype_converters** — [`crate::arrow::ArrowDTypeConverter`]s keyed by [`ExtId`].
///   Consulted when computing the Arrow schema for an extension dtype.
/// - **decoders** / **dtype_readers** — chains for the reverse (Arrow → Vortex) direction.
///   Walked in registration order with user-registered plugins before built-ins.
///
/// The single `canonical_encoder` slot holds the fallback encoder that handles all canonical
/// Vortex encodings. It is set by the default initializer.
#[derive(Debug)]
pub struct ArrowSession {
    encoder_by_encoding: ArrowEncoderByEncodingRegistry,
    encoder_by_extension: ArrowEncoderByExtensionRegistry,
    dtype_converters: ArrowDTypeConverterRegistry,
    /// User-registered decoder chain. Walked before [`ArrowSession::default_decoder`].
    decoders: RwLock<Vec<ArrowDecoderRef>>,
    /// User-registered dtype-reader chain. Walked before [`ArrowSession::default_dtype_reader`].
    dtype_readers: RwLock<Vec<ArrowDTypeReaderRef>>,
    canonical_encoder: RwLock<Option<ArrowEncoderRef>>,
    /// Fallback decoder used after the user chain has declined.
    default_decoder: RwLock<Option<ArrowDecoderRef>>,
    /// Fallback dtype reader used after the user chain has declined.
    default_dtype_reader: RwLock<Option<ArrowDTypeReaderRef>>,
}

impl Default for ArrowSession {
    fn default() -> Self {
        let this = Self {
            encoder_by_encoding: Registry::default(),
            encoder_by_extension: Registry::default(),
            dtype_converters: Registry::default(),
            decoders: RwLock::new(Vec::new()),
            dtype_readers: RwLock::new(Vec::new()),
            canonical_encoder: RwLock::new(None),
            default_decoder: RwLock::new(None),
            default_dtype_reader: RwLock::new(None),
        };
        this.set_canonical_encoder(Arc::new(CanonicalArrowEncoder) as ArrowEncoderRef);
        this.set_default_decoder(Arc::new(CanonicalArrowDecoder) as ArrowDecoderRef);
        this.set_default_dtype_reader(Arc::new(CanonicalArrowDTypeReader) as ArrowDTypeReaderRef);
        // Built-in encoding-keyed encoders for non-canonical optimizations.
        this.register_encoder_for_encoding(
            VarBinArrowEncoder::array_id(),
            Arc::new(VarBinArrowEncoder) as ArrowEncoderRef,
        );
        this.register_encoder_for_encoding(
            ListArrowEncoder::array_id(),
            Arc::new(ListArrowEncoder) as ArrowEncoderRef,
        );

        // Built-in temporal extension plugins.
        let temporal_encoder: ArrowEncoderRef = Arc::new(TemporalArrowEncoder);
        let temporal_converter: ArrowDTypeConverterRef = Arc::new(TemporalArrowDTypeConverter);
        for ext_id in [
            ExtVTable::id(&Date),
            ExtVTable::id(&Time),
            ExtVTable::id(&Timestamp),
        ] {
            this.register_encoder_for_extension(ext_id, Arc::clone(&temporal_encoder));
            this.register_dtype_converter(ext_id, Arc::clone(&temporal_converter));
        }
        this
    }
}

impl SessionVar for ArrowSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ArrowSession {
    /// Register a forward array encoder keyed by encoding [`ArrayId`].
    pub fn register_encoder_for_encoding(
        &self,
        key: impl Into<Id>,
        plugin: impl Into<ArrowEncoderRef>,
    ) {
        self.encoder_by_encoding.register(key, plugin);
    }

    /// Register a forward array encoder keyed by extension [`ExtId`].
    pub fn register_encoder_for_extension(
        &self,
        key: impl Into<Id>,
        plugin: impl Into<ArrowEncoderRef>,
    ) {
        self.encoder_by_extension.register(key, plugin);
    }

    /// Register a dtype converter keyed by extension [`ExtId`].
    pub fn register_dtype_converter(
        &self,
        key: impl Into<Id>,
        plugin: impl Into<ArrowDTypeConverterRef>,
    ) {
        self.dtype_converters.register(key, plugin);
    }

    /// Register a reverse-direction array decoder. Decoders are walked in registration order.
    pub fn register_decoder(&self, plugin: impl Into<ArrowDecoderRef>) {
        self.decoders.write().push(plugin.into());
    }

    /// Register a reverse-direction dtype reader. Readers are walked in registration order.
    pub fn register_dtype_reader(&self, plugin: impl Into<ArrowDTypeReaderRef>) {
        self.dtype_readers.write().push(plugin.into());
    }

    /// Set the canonical (fallback) encoder slot.
    ///
    /// This single plugin must handle every canonical Vortex encoding. Callers replace any
    /// previously-registered canonical encoder.
    pub fn set_canonical_encoder(&self, plugin: impl Into<ArrowEncoderRef>) {
        *self.canonical_encoder.write() = Some(plugin.into());
    }

    /// Set the fallback Arrow → Vortex array decoder.
    ///
    /// Replaces any previously-registered fallback. The fallback runs after the user chain
    /// has declined.
    pub fn set_default_decoder(&self, plugin: impl Into<ArrowDecoderRef>) {
        *self.default_decoder.write() = Some(plugin.into());
    }

    /// Set the fallback Arrow → Vortex dtype reader.
    pub fn set_default_dtype_reader(&self, plugin: impl Into<ArrowDTypeReaderRef>) {
        *self.default_dtype_reader.write() = Some(plugin.into());
    }

    /// Find a forward encoder for the given encoding [`ArrayId`].
    pub fn encoder_for_encoding(&self, id: &ArrayId) -> Option<ArrowEncoderRef> {
        self.encoder_by_encoding.find(id)
    }

    /// Find a forward encoder for the given extension [`ExtId`].
    pub fn encoder_for_extension(&self, id: &ExtId) -> Option<ArrowEncoderRef> {
        self.encoder_by_extension.find(id)
    }

    /// Find a dtype converter for the given extension [`ExtId`].
    pub fn dtype_converter_for(&self, id: &ExtId) -> Option<ArrowDTypeConverterRef> {
        self.dtype_converters.find(id)
    }

    /// Snapshot the current decoder chain (in registration order).
    pub fn decoders(&self) -> Vec<ArrowDecoderRef> {
        self.decoders.read().clone()
    }

    /// Snapshot the current dtype-reader chain (in registration order).
    pub fn dtype_readers(&self) -> Vec<ArrowDTypeReaderRef> {
        self.dtype_readers.read().clone()
    }

    /// The currently-registered canonical encoder, if any.
    pub fn canonical_encoder(&self) -> Option<ArrowEncoderRef> {
        self.canonical_encoder.read().clone()
    }

    /// The currently-registered fallback decoder, if any.
    pub fn default_decoder(&self) -> Option<ArrowDecoderRef> {
        self.default_decoder.read().clone()
    }

    /// The currently-registered fallback dtype reader, if any.
    pub fn default_dtype_reader(&self) -> Option<ArrowDTypeReaderRef> {
        self.default_dtype_reader.read().clone()
    }
}

// --- Forward porcelain (Vortex → Arrow) ---

impl ArrowSession {
    /// Convert a Vortex [`DType`] into the Arrow [`DataType`] this session would emit.
    pub fn to_arrow_data_type(&self, dtype: &DType) -> VortexResult<DataType> {
        if let DType::Extension(ext) = dtype
            && let Some(converter) = self.dtype_converter_for(&ext.id())
        {
            return converter.to_arrow_data_type(ext);
        }
        // Non-extension types and extensions without a converter fall back to the canonical
        // dtype mapping. The shim retains historical hard-coded behavior for callers that
        // don't go through the session.
        dtype.to_arrow_dtype()
    }

    /// Build an Arrow [`Field`] for `dtype` with the given column name.
    pub fn to_arrow_field(&self, name: &str, dtype: &DType) -> VortexResult<Field> {
        if let DType::Extension(ext) = dtype
            && let Some(converter) = self.dtype_converter_for(&ext.id())
        {
            return converter.to_arrow_field(ext, name);
        }
        Ok(Field::new(
            name,
            self.to_arrow_data_type(dtype)?,
            dtype.is_nullable(),
        ))
    }

    /// Build an Arrow [`Schema`] for `struct_dtype`.
    pub fn to_arrow_schema(
        &self,
        struct_dtype: &StructFields,
        nullability: Nullability,
    ) -> VortexResult<Schema> {
        // Defer to the existing top-level helper to preserve extension-metadata handling for
        // Variant fields. The wrapper exists so callers can stop reaching for `DType` directly.
        DType::Struct(struct_dtype.clone(), nullability).to_arrow_schema()
    }

    /// Resolve the Arrow [`DataType`] this session would emit for `array` when no target is
    /// specified. Walks the encoding-keyed encoder, then the extension-keyed encoder, and
    /// finally the canonical encoder.
    pub fn resolve_preferred_arrow_type(&self, array: &ArrayRef) -> VortexResult<DataType> {
        if let Some(plugin) = self.encoder_for_encoding(&array.encoding_id())
            && let Some(t) = plugin.preferred_arrow_type(array, self)?
        {
            return Ok(t);
        }
        if let DType::Extension(ext) = array.dtype()
            && let Some(plugin) = self.encoder_for_extension(&ext.id())
            && let Some(t) = plugin.preferred_arrow_type(array, self)?
        {
            return Ok(t);
        }
        let canonical = self
            .canonical_encoder()
            .ok_or_else(|| vortex_err!("ArrowSession has no canonical encoder registered"))?;
        canonical
            .preferred_arrow_type(array, self)?
            .ok_or_else(|| vortex_err!("canonical encoder produced no preferred Arrow type"))
    }

    /// Convert a Vortex [`ArrayRef`] into an Arrow array.
    ///
    /// `target` selects the Arrow [`DataType`] to emit. Passing [`None`] uses the cheapest
    /// representation the session can produce.
    pub fn to_arrow_array(
        &self,
        array: ArrayRef,
        target: Option<&DataType>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowArrayRef> {
        let target_owned;
        let target = match target {
            Some(t) => t,
            None => {
                target_owned = self.resolve_preferred_arrow_type(&array)?;
                &target_owned
            }
        };

        if let Some(plugin) = self.encoder_for_encoding(&array.encoding_id())
            && let Some(out) = plugin.to_arrow_array(array.clone(), target, ctx)?
        {
            return Ok(out);
        }

        if let DType::Extension(ext) = array.dtype()
            && let Some(plugin) = self.encoder_for_extension(&ext.id())
            && let Some(out) = plugin.to_arrow_array(array.clone(), target, ctx)?
        {
            return Ok(out);
        }

        let canonical = self
            .canonical_encoder()
            .ok_or_else(|| vortex_err!("ArrowSession has no canonical encoder registered"))?;
        match canonical.to_arrow_array(array, target, ctx)? {
            Some(out) => Ok(out),
            None => vortex_bail!("canonical encoder declined Arrow target {target}"),
        }
    }

    /// Convert a Vortex [`ArrayRef`] into an Arrow [`RecordBatch`] matching `schema`.
    pub fn to_arrow_record_batch(
        &self,
        array: ArrayRef,
        schema: &Schema,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RecordBatch> {
        let target = DataType::Struct(schema.fields.clone());
        let arrow = self.to_arrow_array(array, Some(&target), ctx)?;
        Ok(RecordBatch::from(arrow.as_struct()))
    }
}

// --- Reverse porcelain (Arrow → Vortex) ---

impl ArrowSession {
    /// Read a Vortex [`DType`] from an Arrow [`Field`].
    pub fn from_arrow_field(&self, field: &Field) -> VortexResult<DType> {
        for reader in self.dtype_readers() {
            if let Some(dtype) = reader.try_read_dtype(field)? {
                return Ok(dtype);
            }
        }
        if let Some(default) = self.default_dtype_reader()
            && let Some(dtype) = default.try_read_dtype(field)?
        {
            return Ok(dtype);
        }
        vortex_bail!(
            "no ArrowDTypeReader claimed Arrow field with type {}",
            field.data_type()
        )
    }

    /// Read a Vortex [`StructFields`] from an Arrow [`Schema`].
    pub fn from_arrow_schema(&self, schema: &Schema) -> VortexResult<StructFields> {
        let mut entries = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            let dtype = self.from_arrow_field(field)?;
            entries.push((crate::dtype::FieldName::from(field.name().as_str()), dtype));
        }
        Ok(StructFields::from_iter(entries))
    }

    /// Convert an Arrow array into a Vortex [`ArrayRef`].
    pub fn from_arrow_array(
        &self,
        array: &dyn arrow_array::Array,
        field: &Field,
        session: &vortex_session::VortexSession,
    ) -> VortexResult<ArrayRef> {
        for decoder in self.decoders() {
            if let Some(out) = decoder.try_decode(array, field, session)? {
                return Ok(out);
            }
        }
        if let Some(default) = self.default_decoder()
            && let Some(out) = default.try_decode(array, field, session)?
        {
            return Ok(out);
        }
        vortex_bail!(
            "no ArrowDecoder claimed Arrow array of type {}",
            array.data_type()
        )
    }
}

/// Extension trait for accessing the [`ArrowSession`] facet.
pub trait ArrowSessionExt: SessionExt {
    /// Get the Arrow session.
    fn arrow(&self) -> Ref<'_, ArrowSession> {
        self.get::<ArrowSession>()
    }
}

impl<S: SessionExt> ArrowSessionExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Int32Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::PType;
    use crate::session::ArraySession;
    use crate::validity::Validity;

    fn test_session() -> VortexSession {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<ArrowSession>()
    }

    #[test]
    fn forward_canonical_primitive_roundtrips() -> VortexResult<()> {
        let session = test_session();
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        let mut ctx = session.create_execution_ctx();
        let arrow = session.arrow().to_arrow_array(array, None, &mut ctx)?;
        assert_eq!(arrow.data_type(), &DataType::Int32);
        let primitive = arrow.as_primitive::<Int32Type>();
        assert_eq!(primitive.values().as_ref(), &[1, 2, 3]);
        Ok(())
    }

    #[test]
    fn reverse_canonical_primitive_roundtrips() -> VortexResult<()> {
        let session = test_session();
        let arrow_array: Arc<dyn arrow_array::Array> = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let field = Field::new("x", DataType::Int32, false);
        let vortex = session
            .arrow()
            .from_arrow_array(arrow_array.as_ref(), &field, &session)?;
        let primitive = vortex.as_::<crate::arrays::Primitive>();
        assert_eq!(primitive.ptype(), PType::I32);
        Ok(())
    }

    /// Custom decoder used to demonstrate that user-registered plugins run before the default
    /// canonical decoder.
    #[derive(Debug, Default)]
    struct OverrideInt32Decoder;

    impl crate::arrow::ArrowDecoder for OverrideInt32Decoder {
        fn try_decode(
            &self,
            array: &dyn arrow_array::Array,
            _field: &Field,
            _session: &VortexSession,
        ) -> VortexResult<Option<ArrayRef>> {
            if matches!(array.data_type(), DataType::Int32) {
                // Return all-zero array of the same length so the test can observe the override.
                let len = array.len();
                let zeros: Vec<i32> = vec![0; len];
                Ok(Some(
                    PrimitiveArray::new(
                        vortex_buffer::Buffer::<i32>::from(zeros),
                        Validity::NonNullable,
                    )
                    .into_array(),
                ))
            } else {
                Ok(None)
            }
        }
    }

    #[test]
    fn user_registered_decoder_runs_before_default() -> VortexResult<()> {
        let session = test_session();
        session
            .arrow()
            .register_decoder(Arc::new(OverrideInt32Decoder) as ArrowDecoderRef);
        let arrow_array: Arc<dyn arrow_array::Array> = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let field = Field::new("x", DataType::Int32, false);
        let vortex = session
            .arrow()
            .from_arrow_array(arrow_array.as_ref(), &field, &session)?;
        let primitive = vortex.as_::<crate::arrays::Primitive>();
        // Override returned all-zero values, proving it ran instead of the canonical decoder.
        assert_eq!(primitive.as_slice::<i32>(), &[0, 0, 0]);
        Ok(())
    }

    #[test]
    fn temporal_extension_dispatches_to_plugin() -> VortexResult<()> {
        let session = test_session();
        let date_dtype = DType::Extension(
            Date::new(
                crate::extension::datetime::TimeUnit::Days,
                Nullability::NonNullable,
            )
            .erased(),
        );
        let arrow_dt = session.arrow().to_arrow_data_type(&date_dtype)?;
        assert_eq!(arrow_dt, DataType::Date32);
        Ok(())
    }
}
