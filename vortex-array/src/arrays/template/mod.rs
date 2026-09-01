// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scoped symbolic inputs used by lazy data-parallel templates.
//!
//! This is deliberately small: it supports substituting the scalar-function trees produced by a
//! lambda body.  It is not a general protocol for rebuilding arbitrary array encodings.

mod input;
mod instantiate;

pub use input::TemplateInput;
pub use input::TemplateInputArray;
pub use input::TemplateInputArrayExt;
pub use input::TemplateScope;
pub(crate) use instantiate::instantiate;
pub(crate) use instantiate::template_scope;
