// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::Context;

use crate::vtable::DynVTable;

pub type ArrayContext = Context<&'static dyn DynVTable>;
