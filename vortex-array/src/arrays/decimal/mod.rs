// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::DecimalArrayExt;
pub use array::DecimalData;
pub use array::DecimalDataParts;
pub use vtable::DecimalArray;

pub(crate) mod compute;

mod vtable;
pub use compute::rules::DecimalMaskedValidityRule;
pub use vtable::Decimal;

mod utils;
pub use utils::*;

