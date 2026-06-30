// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared SpatialBench table catalog: the single source of truth for the table set, consumed by
//! both [`super::wkb`] data generation and benchmark table registration.

/// A SpatialBench table.
#[derive(Clone, Copy)]
pub enum Table {
    Trip,
    Building,
    Customer,
    Zone,
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
}
