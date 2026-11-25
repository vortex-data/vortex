// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

mod bool;
mod extension;
mod primitive;
mod varbin;

use bool::accumulate_bool;
use extension::accumulate_extension;
use primitive::accumulate_primitive;
use varbin::accumulate_varbin;

use super::BloomPartial;

pub(super) fn accumulate_canonical(
    canonical: &Canonical,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Bool(array) => accumulate_bool(array, partial, ctx)?,
        Canonical::Primitive(array) => accumulate_primitive(array, partial, ctx)?,
        Canonical::VarBinView(array) => accumulate_varbin(array, partial, ctx)?,
        Canonical::Extension(array) => accumulate_extension(array, partial, ctx)?,
        // Nulls are skipped and are not included in any Bloom filter.
        Canonical::Null(_) => {}

        // TODO (joacoc): Pending canonicals
        // TODO (joacoc): Add decimal support once the logical and physical type
        // mismatch is resolved:
        // <https://github.com/vortex-data/vortex/issues/5820>
        Canonical::Decimal(_)
        | Canonical::Struct(_)
        | Canonical::List(_)
        | Canonical::FixedSizeList(_)
        | Canonical::Variant(_)
        | Canonical::Union(_)
        | Canonical::Map(_) => {
            vortex_bail!(
                "Unsupported canonical type for bloom filter: {}",
                canonical.dtype()
            )
        }
    }

    Ok(())
}
