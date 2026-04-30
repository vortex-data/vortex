// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::error::VortexResult;
use vortex_geo::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;

/// Create a new exporter for geospatial data stored in one of the supported spatial formats.
pub(crate) fn new_wkb_exporter(
    array: WellKnownBinaryData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Execute the WKB child into binary
    let values = array
        .wkb_values()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_varbinview();
    crate::exporter::varbinview::new_exporter(values, ctx)
}
