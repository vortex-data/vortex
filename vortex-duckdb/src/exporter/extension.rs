// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_geo::extension::WellKnownBinary;
use vortex_geo::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
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
