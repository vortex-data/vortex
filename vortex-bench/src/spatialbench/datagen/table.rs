// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared SpatialBench table catalog: one source of truth for the base tables, used by both
//! the WKB generation ([`super::wkb`]) and the native geometry conversion ([`super::native`]).

/// A SpatialBench base table.
#[derive(Clone, Copy)]
pub enum Table {
    Trip,
    Building,
    Zone,
}

/// Base tables generated in-process from the scale factor. `Zone` is excluded — it is sourced externally.
pub(crate) const TABLES: &[Table] = &[Table::Trip, Table::Building];

/// A geometry column and the geometry type its WKB bytes decode to.
pub(crate) struct GeometryColumn {
    pub(crate) name: &'static str,
    pub(crate) kind: GeometryKind,
}

/// Geometry types a column can hold. Add a variant (and the matching arm in [`super::native`]) as
/// tables with new geometry types are wired.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GeometryKind {
    Point,
    Polygon,
    /// No native vortex type yet — keep the WKB bytes as `WellKnownBinary` (still surfaces to DuckDB
    /// as `GEOMETRY`). Used for `zone`, whose Overture boundaries include `MultiPolygon`.
    Wkb,
}

impl Table {
    /// File stem under a format directory, e.g. `Trip` → `trip_{part}.parquet`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Table::Trip => "trip",
            Table::Building => "building",
            Table::Zone => "zone",
        }
    }

    /// Columns the wired queries read — the projection applied when building native files.
    pub(crate) fn columns(self) -> &'static [&'static str] {
        match self {
            Table::Trip => &[
                "t_tripkey",
                "t_pickuptime",
                "t_pickuploc",
                "t_dropofftime",
                "t_distance",
                "t_fare",
            ],
            Table::Building => &["b_buildingkey", "b_name", "b_boundary"],
            Table::Zone => &["z_zonekey", "z_name", "z_boundary"],
        }
    }

    /// Geometry columns to decode from WKB to native, with their geometry type. Empty for tables
    /// only used on the WKB lane (DuckDB reads WKB directly; no native conversion needed yet).
    pub(crate) fn geometry_columns(self) -> &'static [GeometryColumn] {
        match self {
            Table::Trip => &[GeometryColumn {
                name: "t_pickuploc",
                kind: GeometryKind::Point,
            }],
            Table::Building => &[GeometryColumn {
                name: "b_boundary",
                kind: GeometryKind::Polygon,
            }],
            Table::Zone => &[GeometryColumn {
                name: "z_boundary",
                kind: GeometryKind::Wkb,
            }],
        }
    }
}
