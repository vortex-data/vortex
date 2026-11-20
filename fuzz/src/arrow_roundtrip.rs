// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzz Arrow field and array round trips for registered extension types.
//!
//! Inputs select an extension family and generate storage with its declared dtype.
//! Most storage uses [`ArbitraryArray`](vortex_array::arrays::arbitrary::ArbitraryArray). JSON and
//! WKB use valid serialized payloads because arbitrary UTF-8 and binary bytes need not represent
//! those logical values. The shared Arrow harness checks each
//! [`ExtensionArray`](vortex_array::arrays::ExtensionArray) in every container. Variant is excluded
//! pending <https://github.com/vortex-data/vortex/issues/11359>.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::Time;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_arrow::test_harness::DefaultExport;
use vortex_arrow::test_harness::check_containers;
use vortex_error::VortexExpect;
use vortex_json::Json;
use vortex_spatial::extension::Dimension;
use vortex_spatial::extension::LineString;
use vortex_spatial::extension::MultiLineString;
use vortex_spatial::extension::MultiPoint;
use vortex_spatial::extension::MultiPolygon;
use vortex_spatial::extension::Point;
use vortex_spatial::extension::Polygon;
use vortex_spatial::extension::Rect;
use vortex_spatial::extension::SpatialMetadata;
use vortex_spatial::extension::WellKnownBinary;
use vortex_tensor::fixed_shape_tensor::FixedShapeTensor;
use vortex_tensor::fixed_shape_tensor::FixedShapeTensorMetadata;
use vortex_tensor::vector::Vector;
use vortex_tensor::vector::VectorMetadata;
use vortex_uuid::Uuid;
use vortex_uuid::UuidMetadata;

use crate::ARROW_ROUND_TRIP_SESSION;

/// The extension family selected by the first byte of a fuzz input.
///
/// Seed tests encode these declaration indices as the first byte. The selection order in
/// [`Arbitrary::arbitrary`] must match this enum so that each seed exercises its named family.
#[derive(Clone, Copy, Debug)]
enum ExtensionKind {
    Uuid,
    Json,
    Vector,
    FixedShapeTensor,
    Date,
    Time,
    Timestamp,
    Point,
    Rect,
    LineString,
    MultiPoint,
    Polygon,
    MultiLineString,
    MultiPolygon,
    WellKnownBinary,
}

impl<'a> Arbitrary<'a> for ExtensionKind {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        u.choose(&[
            Self::Uuid,
            Self::Json,
            Self::Vector,
            Self::FixedShapeTensor,
            Self::Date,
            Self::Time,
            Self::Timestamp,
            Self::Point,
            Self::Rect,
            Self::LineString,
            Self::MultiPoint,
            Self::Polygon,
            Self::MultiLineString,
            Self::MultiPolygon,
            Self::WellKnownBinary,
        ])
        .copied()
    }
}

/// An extension array with valid storage and its expected default-export behavior.
///
/// Input decoding errors are returned to the fuzz engine. Failures to construct the selected
/// dtype or wrap its generated storage are generator defects and panic.
#[derive(Debug)]
pub struct ArrowRoundTrip {
    /// Storage was generated from this extension array's dtype.
    array: ArrayRef,
    /// Retained so the checker can assert both successful and rejected default exports.
    default_export: DefaultExport,
}

impl<'a> Arbitrary<'a> for ArrowRoundTrip {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let kind = u.arbitrary::<ExtensionKind>()?;
        let nullability = u.arbitrary::<bool>()?.into();
        let len = u.int_in_range(0usize..=32)?;

        let dtype = kind.arbitrary_dtype(u, nullability)?;
        let storage = kind.arbitrary_storage(u, dtype.storage_dtype(), len)?;

        Ok(Self {
            array: ExtensionArray::try_new(dtype, storage)
                .vortex_expect("storage was generated from the extension's storage dtype")
                .into_array(),
            default_export: if matches!(kind, ExtensionKind::FixedShapeTensor) {
                DefaultExport::Unsupported
            } else {
                DefaultExport::Supported
            },
        })
    }
}

impl ArrowRoundTrip {
    /// Checks dtype and value preservation in every container, including dictionary and run-end.
    ///
    /// Panics if Arrow transport fails or accepts an unsupported default export.
    #[track_caller]
    pub fn check(self) {
        check_containers(
            self.array,
            self.default_export,
            &mut ARROW_ROUND_TRIP_SESSION.create_execution_ctx(),
        )
        .vortex_expect("Arrow transport must preserve dtype and stored values");
    }
}

impl ExtensionKind {
    /// Generates a dtype whose storage satisfies the selected extension's construction rules.
    fn arbitrary_dtype(
        self,
        u: &mut Unstructured<'_>,
        nullability: Nullability,
    ) -> arbitrary::Result<ExtDTypeRef> {
        let spatial = SpatialMetadata::new(u.arbitrary::<bool>()?.then(|| "EPSG:4326".into()));
        let width = u.int_in_range(0u32..=4)?;
        let ptype = *u.choose(&[PType::F16, PType::F32, PType::F64])?;
        let unit = *u.choose(&[
            TimeUnit::Seconds,
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ])?;

        let dtype = match self {
            Self::Uuid => ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), nullability)
                .vortex_expect("UUID storage is a fixed-size list of 16 non-nullable bytes")
                .erased(),
            Self::Json => ExtDType::try_with_vtable(Json, EmptyMetadata, nullability)
                .vortex_expect("JSON storage was constructed as UTF-8")
                .erased(),
            Self::Vector => ExtDType::try_with_vtable(
                Vector,
                VectorMetadata::try_new(ptype, width)
                    .vortex_expect("vector element type was selected from floats"),
                nullability,
            )
            .vortex_expect("vector storage is a fixed-size list of non-nullable floats")
            .erased(),
            Self::FixedShapeTensor => ExtDType::try_with_vtable(
                FixedShapeTensor,
                FixedShapeTensorMetadata::new(ptype, vec![width as usize]),
                nullability,
            )
            .vortex_expect("tensor shape and fixed-size-list storage have the same width")
            .erased(),
            Self::Date => Date::try_new(
                if u.arbitrary()? {
                    TimeUnit::Days
                } else {
                    TimeUnit::Milliseconds
                },
                nullability,
            )
            .vortex_expect("date unit was selected from days and milliseconds")
            .erased(),
            Self::Time => Time::try_new(unit, nullability)
                .vortex_expect("time unit was selected from seconds through nanoseconds")
                .erased(),
            Self::Timestamp => Timestamp::try_new_with_tz(
                unit,
                u.arbitrary::<bool>()?.then(|| "UTC".into()),
                nullability,
            )
            .vortex_expect("timestamp unit was selected from seconds through nanoseconds")
            .erased(),
            Self::Point => {
                ExtDType::try_with_vtable(Point, spatial.with_dimension(Dimension::Xy), nullability)
                    .vortex_expect("point metadata declares XY coordinates")
                    .erased()
            }
            Self::Rect => {
                ExtDType::try_with_vtable(Rect, spatial.with_dimension(Dimension::Xy), nullability)
                    .vortex_expect("rect metadata declares XY coordinates")
                    .erased()
            }
            Self::LineString => ExtDType::try_with_vtable(
                LineString,
                spatial.with_dimension(Dimension::Xy),
                nullability,
            )
            .vortex_expect("linestring metadata declares XY coordinates")
            .erased(),
            Self::MultiPoint => ExtDType::try_with_vtable(
                MultiPoint,
                spatial.with_dimension(Dimension::Xy),
                nullability,
            )
            .vortex_expect("multipoint metadata declares XY coordinates")
            .erased(),
            Self::Polygon => ExtDType::try_with_vtable(
                Polygon,
                spatial.with_dimension(Dimension::Xy),
                nullability,
            )
            .vortex_expect("polygon metadata declares XY coordinates")
            .erased(),
            Self::MultiLineString => ExtDType::try_with_vtable(
                MultiLineString,
                spatial.with_dimension(Dimension::Xy),
                nullability,
            )
            .vortex_expect("multilinestring metadata declares XY coordinates")
            .erased(),
            Self::MultiPolygon => ExtDType::try_with_vtable(
                MultiPolygon,
                spatial.with_dimension(Dimension::Xy),
                nullability,
            )
            .vortex_expect("multipolygon metadata declares XY coordinates")
            .erased(),
            Self::WellKnownBinary => {
                ExtDType::try_with_vtable(WellKnownBinary, spatial, nullability)
                    .vortex_expect("WKB metadata has no coordinate dimension")
                    .erased()
            }
        };

        Ok(dtype)
    }

    /// Generates storage with valid JSON and WKB payloads for types that constrain their bytes.
    fn arbitrary_storage(
        self,
        u: &mut Unstructured<'_>,
        dtype: &DType,
        len: usize,
    ) -> arbitrary::Result<ArrayRef> {
        let storage = match self {
            Self::Json => {
                let text = u.arbitrary::<i64>()?.to_string();
                let values =
                    VarBinViewArray::from_iter_str(std::iter::repeat_n(text, len)).into_array();

                values
                    .cast(dtype.clone())
                    .vortex_expect("JSON storage only changes the nullability of UTF-8 values")
            }
            Self::WellKnownBinary => {
                // WKB encodes this XY point with a little-endian marker and geometry type 1.
                let mut bytes = vec![1u8];
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&0.0f64.to_le_bytes());
                bytes.extend_from_slice(&1.0f64.to_le_bytes());
                let values =
                    VarBinViewArray::from_iter_bin(std::iter::repeat_n(bytes, len)).into_array();

                values
                    .cast(dtype.clone())
                    .vortex_expect("WKB storage only changes the nullability of binary values")
            }
            _ => {
                ArbitraryArray::arbitrary_with_config(
                    u,
                    &ArbitraryArrayConfig {
                        dtype: Some(dtype.clone()),
                        len: len..=len,
                    },
                )?
                .0
            }
        };

        Ok(storage)
    }
}

#[cfg(test)]
mod tests {
    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;
    use rstest::rstest;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::Time;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_json::Json;
    use vortex_spatial::extension::LineString;
    use vortex_spatial::extension::MultiLineString;
    use vortex_spatial::extension::MultiPoint;
    use vortex_spatial::extension::MultiPolygon;
    use vortex_spatial::extension::Point;
    use vortex_spatial::extension::Polygon;
    use vortex_spatial::extension::Rect;
    use vortex_spatial::extension::WellKnownBinary;
    use vortex_tensor::fixed_shape_tensor::FixedShapeTensor;
    use vortex_tensor::vector::Vector;
    use vortex_uuid::Uuid;

    use super::ArrowRoundTrip;
    use super::ExtensionKind;

    #[track_caller]
    fn check_seeded_round_trip(kind: ExtensionKind, vtable: impl ExtVTable) -> VortexResult<()> {
        // Remaining bytes supply bounded storage draws without coupling the seed to their layout.
        let mut bytes = vec![0u8; 65_536];
        bytes[..5].copy_from_slice(&[
            kind as u8, // Extension family.
            1,          // Nullable storage.
            3,          // Three rows.
            0,          // No spatial CRS.
            2,          // Vector or tensor width.
        ]);
        let input = ArrowRoundTrip::arbitrary(&mut Unstructured::new(&bytes))
            .map_err(|e| vortex_err!("failed to generate seed for family {kind:?}: {e}"))?;

        assert_eq!(input.array.dtype().as_extension().id(), vtable.id());
        assert_eq!(input.array.len(), 3);
        input.check();

        Ok(())
    }

    #[rstest]
    #[case::uuid(ExtensionKind::Uuid, Uuid)]
    #[case::json(ExtensionKind::Json, Json)]
    #[case::vector(ExtensionKind::Vector, Vector)]
    #[case::tensor(ExtensionKind::FixedShapeTensor, FixedShapeTensor)]
    #[case::date(ExtensionKind::Date, Date)]
    #[case::time(ExtensionKind::Time, Time)]
    #[case::timestamp(ExtensionKind::Timestamp, Timestamp)]
    #[case::point(ExtensionKind::Point, Point)]
    #[case::rect(ExtensionKind::Rect, Rect)]
    #[case::line_string(ExtensionKind::LineString, LineString)]
    #[case::multi_point(ExtensionKind::MultiPoint, MultiPoint)]
    #[case::polygon(ExtensionKind::Polygon, Polygon)]
    #[case::multi_line_string(ExtensionKind::MultiLineString, MultiLineString)]
    #[case::multi_polygon(ExtensionKind::MultiPolygon, MultiPolygon)]
    #[case::wkb(ExtensionKind::WellKnownBinary, WellKnownBinary)]
    fn seeded_round_trip(
        #[case] kind: ExtensionKind,
        #[case] vtable: impl ExtVTable,
    ) -> VortexResult<()> {
        check_seeded_round_trip(kind, vtable)
    }
}
