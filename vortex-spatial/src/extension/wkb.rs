// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Well-known binary geometry (`vortex.st.wkb`).
//!
//! [`WellKnownBinary`] stores WKB bytes in binary storage. [`SpatialMetadata`] carries an optional
//! coordinate reference system, while each WKB value encodes its geometry type and dimension.

use std::fmt::Display;
use std::ops::Deref;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::IntoArrow;
use geoarrow::array::WkbViewArray;
use geoarrow::datatypes::WkbType;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExtVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowMetadataPolicy;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;
use wkb::reader::GeometryType;

use crate::extension::SpatialMetadata;
use crate::extension::geoarrow_metadata;
use crate::extension::spatial_metadata_from_arrow;

/// A typed handle to an [`ExtensionArray`] that contains WKB-encoded data.
///
/// You can construct this safely using `WellKnownBinaryData::try_from(ExtensionArray)`.
#[derive(Debug, Clone)]
pub struct WellKnownBinaryData {
    ext: ExtensionArray,
}

impl WellKnownBinaryData {
    /// A reference to the array that holds the Well-Known Binary scalar values.
    pub fn wkb_values(&self) -> &ArrayRef {
        self.ext.storage_array()
    }

    /// A reference to the [spatial metadata][SpatialMetadata].
    pub fn spatial_metadata(&self) -> &SpatialMetadata {
        self.ext
            .dtype()
            .as_extension()
            .metadata::<WellKnownBinary>()
    }
}

impl TryFrom<ExtensionArray> for WellKnownBinaryData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<WellKnownBinary>(),
            "array extension dtype {} is not a WKB",
            ext.ext_dtype()
        );

        Ok(Self { ext })
    }
}

/// An [extension type][ExtVTable] for OGC Well-known Binary (WKB) data format.
///
/// This is one of the most common formats for sharing of geometry data between analytic systems,
/// used by DuckDB, PostGIS and GeoParquet.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct WellKnownBinary;

/// A reference to a value of well-known binary encoded geometry data.
///
/// Interpreting this value is dependent on the [spatial metadata][SpatialMetadata] in the extension
/// schema for the array where this scalar is taken from.
pub struct Wkb<'a>(wkb::reader::Wkb<'a>);

impl<'a> Wkb<'a> {
    /// Attempt to decode a well-known binary value from a byte slice.
    ///
    /// This will not cause any data allocations or copies, but it will perform a one-pass
    /// validation on the structure of the WKB.
    pub fn try_from_bytes(bytes: &'a [u8]) -> VortexResult<Self> {
        wkb::reader::Wkb::try_new(bytes)
            .map_err(|e| vortex_err!("failed parsing WKB: {e}"))
            .map(Wkb)
    }
}

impl<'a> Display for Wkb<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let geometry_kind = match self.0.geometry_type() {
            GeometryType::Point => "point",
            GeometryType::LineString => "linestring",
            GeometryType::Polygon => "polygon",
            GeometryType::MultiPoint => "multipoint",
            GeometryType::MultiLineString => "multilinestring",
            GeometryType::MultiPolygon => "multipolygon",
            GeometryType::GeometryCollection => "geometrycollection",
            _ => "unknown",
        };
        // TODO(aduffy): make this more useful
        write!(f, "WKB({geometry_kind})")
    }
}

impl<'a> Deref for Wkb<'a> {
    type Target = wkb::reader::Wkb<'a>;

    fn deref(&self) -> &wkb::reader::Wkb<'a> {
        &self.0
    }
}

impl ExtVTable for WellKnownBinary {
    type Metadata = SpatialMetadata;
    type StoredMetadata = super::SpatialStoredMetadata;

    type NativeValue<'a> = Wkb<'a>;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.st.wkb");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.serialize())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        super::SpatialStoredMetadata::deserialize(metadata)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(SpatialMetadata::from_stored_metadata(stored_metadata, None))
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Binary(Nullability::NonNullable)
    }

    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        if let Some(dimension) = metadata.dimension {
            vortex_bail!(
                "vortex.st.wkb metadata must not have a coordinate dimension, got {dimension:?}"
            );
        }

        Ok(())
    }

    fn validate_array(
        ext_dtype: &ExtDType<Self>,
        storage: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(storage.dtype(), ext_dtype.storage_dtype());
        let array = match storage.clone().execute::<Columnar>(ctx)? {
            Columnar::Constant(array) => {
                if !array.is_empty()
                    && let Some(value) = array.scalar().value()
                {
                    Self::validate_scalar_value(ext_dtype, value)?;
                }
                return Ok(());
            }
            Columnar::Canonical(Canonical::VarBinView(array)) => array,
            Columnar::Canonical(_) => vortex_bail!("WKB storage must be binary"),
        };
        let valid = array.as_ref().validity()?.execute_mask(array.len(), ctx)?;
        if valid.all_false() {
            return Ok(());
        }
        let array = if array.as_ref().is_host() {
            array
        } else {
            let buffers = array
                .as_ref()
                .buffer_handles()
                .into_iter()
                .map(|buffer| buffer.try_into_host_sync().map(BufferHandle::new_host))
                .collect::<VortexResult<Vec<_>>>()?;
            // SAFETY: each buffer is a byte-for-byte copy of the original buffer.
            unsafe { array.into_array().with_buffers(buffers)? }.execute::<VarBinViewArray>(ctx)?
        };

        let views = array.resolved_views();
        for (view, valid) in views.views().iter().zip(valid.iter()) {
            if valid {
                Wkb::try_from_bytes(views.view_bytes(view))?;
            }
        }

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Wkb::try_from_bytes(storage_value.as_binary().as_slice())
    }
}

static ARROW_WKB: CachedId = CachedId::new(WkbType::NAME);

impl ArrowExtVTable for WellKnownBinary {
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::ArrowNative
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_WKB
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let spatial_metadata = ext_dtype.metadata();

        let mut field = session.to_arrow_field(name, ext_dtype.storage_dtype())?;
        field.try_with_extension_type(wkb_type(spatial_metadata))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let Ok(wkb_meta) = target.try_extension_type::<WkbType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();

        let storage_field = Field::new(
            String::new(),
            target.data_type().clone(),
            target.is_nullable(),
        );
        let arrow_storage =
            storage.execute_arrow_with_options(Some(&storage_field), options, ctx)?;

        // Round-trip through the GeoArrow WKB array types: this validates that the storage
        // is a binary-family Arrow array and produces the canonical physical representation
        // expected for a `WkbType` extension field.
        let arrow_ref: ArrowArrayRef = match target.data_type() {
            DataType::Binary => Arc::new(
                GenericWkbArray::<i32>::try_from((arrow_storage.as_ref(), wkb_meta))
                    .map_err(|e| vortex_err!("failed to construct WkbArray: {e}"))?
                    .into_arrow(),
            ),
            DataType::LargeBinary => Arc::new(
                GenericWkbArray::<i64>::try_from((arrow_storage.as_ref(), wkb_meta))
                    .map_err(|e| vortex_err!("failed to construct LargeWkbArray: {e}"))?
                    .into_arrow(),
            ),
            DataType::BinaryView => Arc::new(
                WkbViewArray::try_from((arrow_storage.as_ref(), wkb_meta))
                    .map_err(|e| vortex_err!("failed to construct WkbViewArray: {e}"))?
                    .into_arrow(),
            ),
            _ => unreachable!("target data type was validated above"),
        };

        Ok(ArrowExport::Exported(arrow_ref))
    }

    fn from_arrow_field(
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
        let wkb_meta = field.try_extension_type::<WkbType>()?;

        Ok(Some(ExtDType::try_with_vtable(
            WellKnownBinary,
            spatial_metadata(&wkb_meta),
            field.is_nullable().into(),
        )?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        _field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        session
            .from_arrow_storage(array, &ext_dtype.clone().erased())
            .map(ArrowImport::Imported)
    }
}

fn wkb_type(spatial_metadata: &SpatialMetadata) -> WkbType {
    WkbType::new(geoarrow_metadata(spatial_metadata))
}

fn spatial_metadata(wkb_type: &WkbType) -> SpatialMetadata {
    spatial_metadata_from_arrow(wkb_type.metadata())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::compute::conformance::extension::test_ext_vtable_conformance;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::extension::SpatialMetadata;
    use crate::extension::WellKnownBinary;

    // Little-endian WKB for POINT (0 0).
    const POINT: &[u8] = &[
        1, 1, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
    ];

    #[rstest]
    #[case::valid(Some(POINT), true)]
    #[case::truncated(Some(&POINT[..5]), false)]
    #[case::null(None, true)]
    fn validates_wkb_array(
        #[case] value: Option<&[u8]>,
        #[case] accepted: bool,
    ) -> VortexResult<()> {
        let storage = VarBinViewArray::from_iter_nullable_bin([value]).into_array();
        let dtype = ExtDType::<WellKnownBinary>::try_new(
            SpatialMetadata::default(),
            Nullability::Nullable,
        )?;
        let mut ctx = VortexSession::empty().create_execution_ctx();

        let result = WellKnownBinary::validate_array(&dtype, &storage, &mut ctx);
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        test_ext_vtable_conformance(&dtype, &storage, &mut ctx);

        Ok(())
    }
}
