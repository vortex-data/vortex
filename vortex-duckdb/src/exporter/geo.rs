// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::VarBinViewArray;
use vortex::error::VortexResult;
use vortex_geo::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::varbinview::new_exporter;

/// Create a new exporter for geospatial data stored as Well-Known Binary (WKB) format.
pub(crate) fn new_wkb_exporter(
    array: WellKnownBinaryData,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let values = array.wkb_values().clone().execute::<VarBinViewArray>(ctx)?;
    new_exporter(values, ctx)
}
