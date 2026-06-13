// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpr;
use vortex_array::expr::placeholder::Placeholder;
use vortex_array::expr::placeholder::PlaceholderId;
use vortex_array::expr::placeholder::PlaceholderRef;
use vortex_session::registry::CachedId;

static ROW_IDX_DTYPE: LazyLock<DType> =
    LazyLock::new(|| DType::Primitive(PType::U64, Nullability::NonNullable));

/// Placeholder for the row index of the current scan scope.
#[derive(Clone, Debug)]
pub struct RowIdx;

impl Placeholder for RowIdx {
    type Payload = ();

    fn id(&self) -> PlaceholderId {
        static ID: CachedId = CachedId::new("vortex.row_idx");
        *ID
    }

    fn dtype(&self) -> &DType {
        &ROW_IDX_DTYPE
    }

    fn display_name(&self) -> &str {
        "row_idx"
    }

    fn payload(&self) -> &Self::Payload {
        &()
    }
}

/// Returns the row-index placeholder reference.
pub fn row_idx_ref() -> PlaceholderRef {
    PlaceholderRef::new(RowIdx)
}

/// Returns the row-index placeholder expression.
pub fn row_idx() -> BoundExpr {
    BoundExpr::Placeholder(row_idx_ref())
}
