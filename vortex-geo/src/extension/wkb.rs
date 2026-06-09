// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::WkbType;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrow::ArrowExport;
use vortex_array::arrow::ArrowExportVTable;
use vortex_array::arrow::ArrowImport;
use vortex_array::arrow::ArrowImportVTable;
use vortex_array::arrow::ArrowSession;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;
use wkb::reader::GeometryType;

use crate::extension::GeoMetadata;

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

    /// A reference to the [geospatial metadata][GeoMetadata].
    pub fn geo_metadata(&self) -> &GeoMetadata {
        self.ext
            .dtype()
            .as_extension()
            .metadata::<WellKnownBinary>()
    }
}

impl TryFrom<ExtensionArray> for WellKnownBinaryData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        if !ext.ext_dtype().is::<WellKnownBinary>() {
            vortex_bail!("array extension dtype {} is not a WKB", ext.ext_dtype());
        }

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
/// Interpreting this value is dependent on the [geospatial metadata][GeoMetadata] in the extension
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
    type Metadata = GeoMetadata;

    type NativeValue<'a> = Wkb<'a>;

    fn id(&self) -> ExtId {
        ExtId::new_static("vortex.geo.wkb")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            ext_dtype.storage_dtype().is_binary(),
            "geo.wkb must have binary storage type, was {}",
            ext_dtype.storage_dtype()
        );

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

impl ArrowExportVTable for WellKnownBinary {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_WKB
    }

    fn vortex_id(&self) -> Id {
        self.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let ext_type = dtype.as_extension();
        let geo_metadata = ext_type.metadata::<WellKnownBinary>();

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(wkb_type(geo_metadata))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_wkb = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<WellKnownBinary>())
            .unwrap_or(false);
        if !is_wkb {
            return Ok(ArrowExport::Unsupported(array));
        }

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
        let session = ctx.session().clone();
        let arrow_storage = session
            .arrow()
            .execute_arrow(storage, Some(&storage_field), ctx)?;

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
}

impl ArrowImportVTable for WellKnownBinary {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_WKB
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let Ok(wkb_meta) = field.try_extension_type::<WkbType>() else {
            return Ok(None);
        };

        let storage_dtype = DType::Binary(field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(WellKnownBinary, geo_metadata(&wkb_meta), storage_dtype)?
                .erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
    ) -> VortexResult<ArrowImport> {
        let Some(ext_dtype) = dtype.as_extension_opt() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !ext_dtype.is::<WellKnownBinary>()
            || field.try_extension_type::<WkbType>().is_err()
            || !matches!(
                array.data_type(),
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView
            )
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = ArrayRef::from_arrow(array.as_ref(), field.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::new(ext_dtype.clone(), storage).into_array(),
        ))
    }
}

fn wkb_type(geo_metadata: &GeoMetadata) -> WkbType {
    let metadata = Metadata::new(
        geo_metadata
            .crs
            .as_ref()
            .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
            .unwrap_or_default(),
        None,
    );
    WkbType::new(Arc::new(metadata))
}

fn geo_metadata(wkb_type: &WkbType) -> GeoMetadata {
    let crs = wkb_type.metadata().crs().crs_value().map(|value| {
        // `Crs::from_unknown_crs_type` stores the user's string verbatim as a JSON string
        // value, so prefer the raw string when available to round-trip cleanly. For other
        // CRS encodings (PROJJSON object, etc.), fall back to the JSON-encoded form.
        value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string())
    });
    GeoMetadata { crs }
}
