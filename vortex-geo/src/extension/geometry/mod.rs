// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Geometry`] extension type (`vortex.geo.geometry`): one logical type for all
//! GeoArrow-native geometry kinds.

pub(crate) mod coordinate;

use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use self::coordinate::Coordinate;
use self::coordinate::Dimension;
use self::coordinate::coordinate_dimension;
use self::coordinate::coordinate_from_struct;
use super::GeoMetadata;
use super::GeometryKind;

/// The `vortex.geo.geometry` extension type: native geometry of a single [`GeometryKind`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Geometry;

impl Geometry {
    /// The extension dtype for a column of `kind` geometries over `storage`.
    pub fn dtype(
        kind: GeometryKind,
        crs: Option<String>,
        storage: DType,
    ) -> VortexResult<ExtDType<Geometry>> {
        ExtDType::try_new(
            GeoMetadata {
                crs,
                geometry_type: kind as i32,
            },
            storage,
        )
    }

    /// The [`GeometryKind`] of a geometry-typed `dtype`; errors on non-geometry dtypes.
    pub fn kind_of(dtype: &DType) -> VortexResult<GeometryKind> {
        let Some(ext) = dtype.as_extension_opt() else {
            vortex_bail!("expected a geometry column, was {dtype}");
        };
        vortex_ensure!(
            ext.is::<Geometry>(),
            "expected a geometry column, was {dtype}"
        );
        ext.metadata::<Geometry>().kind()
    }
}

/// Validate that `storage` is `kind`'s GeoArrow layout.
///
/// Only point columns are supported end to end; other kinds are rejected here until their
/// scalar unpacking and kernels exist, so that a valid dtype never produces arrays whose
/// scalars fail to unpack.
fn validate_storage(kind: GeometryKind, storage: &DType) -> VortexResult<Dimension> {
    match kind {
        GeometryKind::Unspecified => {
            vortex_bail!("geometry kind must be specified; mixed columns are not yet supported")
        }
        GeometryKind::Point => coordinate_dimension(storage),
        kind => vortex_bail!("{kind} geometry columns are not yet supported"),
    }
}

impl ExtVTable for Geometry {
    type Metadata = GeoMetadata;
    type NativeValue<'a> = GeometryValue;

    fn id(&self) -> ExtId {
        ExtId::new_static("vortex.geo.geometry")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        validate_storage(ext_dtype.metadata().kind()?, ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<GeometryValue> {
        match ext_dtype.metadata().kind()? {
            GeometryKind::Point => {
                let storage = Scalar::try_new(
                    ext_dtype.storage_dtype().clone(),
                    Some(storage_value.clone()),
                )?;
                coordinate_from_struct(&storage).map(GeometryValue::Point)
            }
            kind => vortex_bail!("unpacking {kind} scalars is not yet implemented"),
        }
    }
}

/// A decoded native geometry scalar value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GeometryValue {
    /// A single coordinate.
    Point(Coordinate),
    /// A line of coordinates.
    LineString(Vec<Coordinate>),
    /// An outer ring plus zero or more interior rings (holes).
    Polygon(Vec<Vec<Coordinate>>),
    /// A collection of points.
    MultiPoint(Vec<Coordinate>),
    /// A collection of linestrings.
    MultiLineString(Vec<Vec<Coordinate>>),
    /// A collection of polygons.
    MultiPolygon(Vec<Vec<Vec<Coordinate>>>),
}

/// The WKT dimension tag (`" Z"`, `" M"`, `" ZM"`, or empty), read from `first`.
fn dimension_tag(first: Option<&Coordinate>) -> &'static str {
    match first.map(|coordinate| (coordinate.z, coordinate.m)) {
        None | Some((None, None)) => "",
        Some((Some(_), None)) => " Z",
        Some((None, Some(_))) => " M",
        Some((Some(_), Some(_))) => " ZM",
    }
}

/// Write the bare ordinates of one coordinate: `x y {z} {m}`.
fn fmt_ordinates(f: &mut Formatter<'_>, coordinate: &Coordinate) -> std::fmt::Result {
    write!(f, "{} {}", coordinate.x, coordinate.y)?;
    if let Some(z) = coordinate.z {
        write!(f, " {z}")?;
    }
    if let Some(m) = coordinate.m {
        write!(f, " {m}")?;
    }
    Ok(())
}

/// Write `(…, …)`, formatting each element with `item`.
fn fmt_seq<T>(
    f: &mut Formatter<'_>,
    items: &[T],
    item: impl Fn(&mut Formatter<'_>, &T) -> std::fmt::Result,
) -> std::fmt::Result {
    write!(f, "(")?;
    for (idx, value) in items.iter().enumerate() {
        if idx > 0 {
            write!(f, ", ")?;
        }
        item(f, value)?;
    }
    write!(f, ")")
}

/// Write a WKT body: ` EMPTY` if empty, otherwise ` {tag} (…)`.
fn fmt_body<T>(
    f: &mut Formatter<'_>,
    items: &[T],
    first: Option<&Coordinate>,
    item: impl Fn(&mut Formatter<'_>, &T) -> std::fmt::Result,
) -> std::fmt::Result {
    if items.is_empty() {
        return write!(f, " EMPTY");
    }
    write!(f, "{} ", dimension_tag(first))?;
    fmt_seq(f, items, item)
}

impl Display for GeometryValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let coords =
            |f: &mut Formatter<'_>, line: &Vec<Coordinate>| fmt_seq(f, line, fmt_ordinates);
        let rings = |f: &mut Formatter<'_>, rings: &Vec<Vec<Coordinate>>| fmt_seq(f, rings, coords);
        match self {
            GeometryValue::Point(coordinate) => Display::fmt(coordinate, f),
            GeometryValue::LineString(line) => {
                write!(f, "LINESTRING")?;
                fmt_body(f, line, line.first(), fmt_ordinates)
            }
            GeometryValue::MultiPoint(points) => {
                write!(f, "MULTIPOINT")?;
                fmt_body(f, points, points.first(), fmt_ordinates)
            }
            GeometryValue::Polygon(polygon) => {
                write!(f, "POLYGON")?;
                let first = polygon.first().and_then(|ring| ring.first());
                fmt_body(f, polygon, first, coords)
            }
            GeometryValue::MultiLineString(lines) => {
                write!(f, "MULTILINESTRING")?;
                let first = lines.first().and_then(|line| line.first());
                fmt_body(f, lines, first, coords)
            }
            GeometryValue::MultiPolygon(polygons) => {
                write!(f, "MULTIPOLYGON")?;
                let first = polygons
                    .first()
                    .and_then(|polygon| polygon.first())
                    .and_then(|ring| ring.first());
                fmt_body(f, polygons, first, rings)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Geometry;
    use super::GeometryValue;
    use super::coordinate::Coordinate;
    use super::coordinate::Dimension;
    use super::coordinate::coordinate_dimension;
    use super::coordinate::coordinate_from_scalar;
    use crate::extension::GeometryKind;

    /// A coordinate storage dtype with the given field names, non-nullable `f64` per field.
    fn coordinate_dtype(names: &[&'static str]) -> DType {
        let fields = std::iter::repeat_n(
            DType::Primitive(PType::F64, Nullability::NonNullable),
            names.len(),
        )
        .collect::<Vec<_>>();
        DType::Struct(
            StructFields::new(FieldNames::from(names), fields),
            Nullability::NonNullable,
        )
    }

    /// `storage` wrapped in `depth` non-nullable `List` layers.
    fn nested_list(storage: DType, depth: usize) -> DType {
        let mut dtype = storage;
        for _ in 0..depth {
            dtype = DType::List(Arc::new(dtype), Nullability::NonNullable);
        }
        dtype
    }

    /// All four GeoArrow dimensions validate as point storage and round-trip via field names.
    #[test]
    fn point_validates_every_dimension() -> VortexResult<()> {
        let cases = [
            (Dimension::Xy, ["x", "y"].as_slice()),
            (Dimension::Xyz, ["x", "y", "z"].as_slice()),
            (Dimension::Xym, ["x", "y", "m"].as_slice()),
            (Dimension::Xyzm, ["x", "y", "z", "m"].as_slice()),
        ];
        for (dim, names) in cases {
            let storage = coordinate_dtype(names);
            assert_eq!(coordinate_dimension(&storage)?, dim);
            Geometry::dtype(GeometryKind::Point, Some("EPSG:4326".to_string()), storage)?;
        }
        Ok(())
    }

    /// Non-point kinds are rejected at dtype construction — even with their correct GeoArrow
    /// layout — until their scalar unpacking and kernels exist.
    #[rstest]
    #[case(GeometryKind::LineString, 1)]
    #[case(GeometryKind::MultiPoint, 1)]
    #[case(GeometryKind::Polygon, 2)]
    #[case(GeometryKind::MultiLineString, 2)]
    #[case(GeometryKind::MultiPolygon, 3)]
    fn non_point_kinds_are_rejected(#[case] kind: GeometryKind, #[case] depth: usize) {
        let storage = nested_list(coordinate_dtype(&["x", "y"]), depth);
        assert!(Geometry::dtype(kind, None, storage).is_err());
    }

    /// Construction rejects non-struct storage, non-coordinate fields, and the `Unspecified`
    /// kind.
    #[test]
    fn rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(Geometry::dtype(GeometryKind::Point, None, primitive).is_err());

        let wrong_fields = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter(vec![0.0f64]).into_array()),
            ("b", PrimitiveArray::from_iter(vec![0.0f64]).into_array()),
        ])?
        .into_array();
        assert!(Geometry::dtype(GeometryKind::Point, None, wrong_fields.dtype().clone()).is_err());

        assert!(
            Geometry::dtype(
                GeometryKind::Unspecified,
                None,
                coordinate_dtype(&["x", "y"])
            )
            .is_err()
        );
        Ok(())
    }

    /// A point column round-trips through scalar execution back to its coordinates.
    #[test]
    fn point_unpacks_coordinates() -> VortexResult<()> {
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let storage = StructArray::from_fields(&[
            (
                "x",
                PrimitiveArray::from_iter(vec![1.0f64, -111.7610]).into_array(),
            ),
            (
                "y",
                PrimitiveArray::from_iter(vec![2.0f64, 34.8697]).into_array(),
            ),
        ])?
        .into_array();
        let dtype = Geometry::dtype(
            GeometryKind::Point,
            Some("EPSG:4326".to_string()),
            storage.dtype().clone(),
        )?;
        let points = ExtensionArray::new(dtype.erased(), storage).into_array();

        assert_eq!(
            coordinate_from_scalar(&points.execute_scalar(0, &mut ctx)?)?,
            Coordinate::xy(1.0, 2.0)
        );
        assert_eq!(
            coordinate_from_scalar(&points.execute_scalar(1, &mut ctx)?)?,
            Coordinate::xy(-111.7610, 34.8697)
        );
        Ok(())
    }

    /// `GeometryValue` displays as WKT for every kind, including dimension tags and `EMPTY`.
    #[test]
    fn display_is_wkt() {
        let xy = Coordinate::xy;
        assert_eq!(GeometryValue::Point(xy(1.0, 2.0)).to_string(), "POINT(1 2)");
        assert_eq!(
            GeometryValue::LineString(vec![xy(0.0, 0.0), xy(1.0, 1.0)]).to_string(),
            "LINESTRING (0 0, 1 1)"
        );
        assert_eq!(
            GeometryValue::LineString(vec![]).to_string(),
            "LINESTRING EMPTY"
        );
        assert_eq!(
            GeometryValue::MultiPoint(vec![xy(0.0, 0.0), xy(1.0, 1.0)]).to_string(),
            "MULTIPOINT (0 0, 1 1)"
        );
        assert_eq!(
            GeometryValue::Polygon(vec![vec![xy(0.0, 0.0), xy(1.0, 0.0), xy(0.0, 0.0)]])
                .to_string(),
            "POLYGON ((0 0, 1 0, 0 0))"
        );
        assert_eq!(
            GeometryValue::MultiLineString(vec![vec![xy(0.0, 0.0), xy(1.0, 1.0)]]).to_string(),
            "MULTILINESTRING ((0 0, 1 1))"
        );
        assert_eq!(
            GeometryValue::MultiPolygon(vec![vec![vec![xy(0.0, 0.0), xy(1.0, 0.0), xy(0.0, 0.0)]]])
                .to_string(),
            "MULTIPOLYGON (((0 0, 1 0, 0 0)))"
        );
        let zm = Coordinate {
            x: 1.0,
            y: 2.0,
            z: Some(3.0),
            m: Some(4.0),
        };
        assert_eq!(
            GeometryValue::LineString(vec![zm, zm]).to_string(),
            "LINESTRING ZM (1 2 3 4, 1 2 3 4)"
        );
        let z = Coordinate { m: None, ..zm };
        assert_eq!(
            GeometryValue::LineString(vec![z]).to_string(),
            "LINESTRING Z (1 2 3)"
        );
        let m = Coordinate { z: None, ..zm };
        assert_eq!(
            GeometryValue::MultiPoint(vec![m]).to_string(),
            "MULTIPOINT M (1 2 4)"
        );
    }
}
