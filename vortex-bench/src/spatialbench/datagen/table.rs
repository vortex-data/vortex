// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared SpatialBench table catalog: one source of truth for the base tables, used by both
//! the WKB generation ([`super::wkb`]) and the native geometry conversion ([`super::native`]).

/// A SpatialBench base table.
///
/// Add a variant here (plus its [`Table::name`], [`Table::columns`], and [`Table::geometry_columns`]
/// arms, and the generator arm in [`super::wkb`]) as queries start needing the dimension tables
/// (customer, driver, vehicle, zone, building).
#[derive(Clone, Copy)]
pub enum Table {
    Trip,
}

/// Every base table. WKB generation emits all of them; native conversion handles those with
/// geometry columns.
pub(crate) const TABLES: &[Table] = &[Table::Trip];

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
}

impl Table {
    /// File stem under a format directory, e.g. `Trip` → `trip_{part}.parquet`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Table::Trip => "trip",
        }
    }

    /// Columns the wired queries read — the projection applied when building native files.
    pub(crate) fn columns(self) -> &'static [&'static str] {
        match self {
            Table::Trip => &["t_tripkey", "t_pickuptime", "t_pickuploc"],
        }
    }

    /// Geometry columns to decode from WKB to native, with their geometry type.
    pub(crate) fn geometry_columns(self) -> &'static [GeometryColumn] {
        match self {
            Table::Trip => &[GeometryColumn {
                name: "t_pickuploc",
                kind: GeometryKind::Point,
            }],
        }
    }
}
