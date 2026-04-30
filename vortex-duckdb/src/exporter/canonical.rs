// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::extension::datetime::AnyTemporal;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_geo::extension::WellKnownBinary;
use vortex_geo::extension::WellKnownBinaryData;

use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::all_invalid;
use crate::exporter::bool;
use crate::exporter::decimal;
use crate::exporter::fixed_size_list;
use crate::exporter::geo;
use crate::exporter::list_view;
use crate::exporter::primitive;
use crate::exporter::struct_;
use crate::exporter::temporal;
use crate::exporter::varbinview;

pub(crate) fn new_exporter(
    array: ArrayRef,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    match array.execute::<Canonical>(ctx)? {
        Canonical::Null(_) => Ok(all_invalid::new_exporter()),
        Canonical::Bool(array) => bool::new_exporter(array, ctx),
        Canonical::Primitive(array) => primitive::new_exporter(array, ctx),
        Canonical::Decimal(array) => decimal::new_exporter(array, ctx),
        Canonical::VarBinView(array) => varbinview::new_exporter(array, ctx),
        Canonical::List(array) => list_view::new_exporter(array, cache, ctx),
        Canonical::FixedSizeList(array) => fixed_size_list::new_exporter(array, cache, ctx),
        Canonical::Struct(array) => struct_::new_exporter(array, cache, ctx),
        Canonical::Extension(ext) => {
            if ext.ext_dtype().is::<AnyTemporal>() {
                return temporal::new_exporter(TemporalArray::try_from(ext)?, ctx);
            }

            if ext.ext_dtype().is::<WellKnownBinary>() {
                return geo::new_wkb_exporter(WellKnownBinaryData::try_from(ext)?, ctx);
            }

            vortex_bail!("no non-temporal extension exporter")
        }
        Canonical::Variant(_) => {
            vortex_bail!("Variant arrays can't be exported to DuckDB")
        }
    }
}
