// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod execution;
pub mod scalar;
mod session;
mod vtable;

use arcref::ArcRef;
pub use execution::*;
pub use session::*;
pub use vtable::*;

pub type FunctionId = ArcRef<str>;
