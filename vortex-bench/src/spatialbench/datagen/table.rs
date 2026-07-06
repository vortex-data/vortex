// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared SpatialBench table catalog: one source of truth for the base tables, used by both the
//! WKB generation ([`super::wkb`]) and the native geometry conversion ([`super::native`]).

/// A SpatialBench table.
#[derive(Clone, Copy)]
pub enum Table {
    Trip,
    Building,
    Customer,
    Zone,
}

/// A geometry column and the geometry type its WKB bytes decode to.
pub(crate) struct GeometryColumn {
    pub(crate) name: &'static str,
    pub(crate) kind: GeometryKind,
}

/// Geometry types a column can decode to on the native lane.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GeometryKind {
    Point,
    Polygon,
    MultiPolygon,
}

impl Table {
    /// Every SpatialBench table, in registration order.
    pub(crate) const ALL: [Table; 4] = [Table::Trip, Table::Building, Table::Customer, Table::Zone];

    /// File stem under a format directory, e.g. `Trip` → `trip_{part}.parquet`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Table::Trip => "trip",
            Table::Building => "building",
            Table::Customer => "customer",
            Table::Zone => "zone",
        }
    }

    /// Whether this table is generated in-process from the scale factor. `Zone` is sourced externally.
    pub(crate) fn is_generated(self) -> bool {
        !matches!(self, Table::Zone)
    }

    /// Geometry columns to decode from WKB to native, with their geometry type. Empty for tables with
    /// no geometry (e.g. `customer`).
    pub(crate) fn geometry_columns(self) -> &'static [GeometryColumn] {
        match self {
            Table::Trip => &[
                GeometryColumn {
                    name: "t_pickuploc",
                    kind: GeometryKind::Point,
                },
                GeometryColumn {
                    name: "t_dropoffloc",
                    kind: GeometryKind::Point,
                },
            ],
            Table::Building => &[GeometryColumn {
                name: "b_boundary",
                kind: GeometryKind::Polygon,
            }],
            Table::Customer => &[],
            Table::Zone => &[GeometryColumn {
                name: "z_boundary",
                kind: GeometryKind::MultiPolygon,
            }],
        }
    }
}
