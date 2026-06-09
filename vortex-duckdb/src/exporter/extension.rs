// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::extension::datetime::AnyTemporal;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_geo::extension::WellKnownBinary;
use vortex_geo::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::geo;
use crate::exporter::temporal;

pub(crate) fn new_exporter(
    ext: ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if ext.ext_dtype().is::<AnyTemporal>() {
        return temporal::new_exporter(TemporalArray::try_from(ext)?, ctx);
    }

    if ext.ext_dtype().is::<WellKnownBinary>() {
        return geo::new_wkb_exporter(WellKnownBinaryData::try_from(ext)?, ctx);
    }

    vortex_bail!("no non-temporal extension exporter")
}
