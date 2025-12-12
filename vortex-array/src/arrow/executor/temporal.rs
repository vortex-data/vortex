// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_session::VortexSession;
use crate::ArrayRef;

pub(super) fn to_arrow_timestamp(
    array: &ArrayRef,
    nullability: Nullability,
    session: &VortexSession,
)