// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_session::registry::Id;

use crate::dtype::DType;

/// A placeholder identity.
pub type PlaceholderId = Id;

/// Typed placeholder definition.
pub trait Placeholder: 'static + Send + Sync + Debug {
    /// Returns the globally unique placeholder id.
    fn id(&self) -> PlaceholderId;

    /// Returns the dtype this placeholder evaluates to.
    fn dtype(&self) -> &DType;

    /// Returns the short display name used by SQL formatting.
    fn display_name(&self) -> &str;
}

trait DynPlaceholder: 'static + Send + Sync {
    fn as_any(&self) -> &(dyn Any + Send + Sync);
    fn id(&self) -> PlaceholderId;
    fn dtype(&self) -> &DType;
    fn display_name(&self) -> &str;
    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<P: Placeholder> DynPlaceholder for P {
    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn id(&self) -> PlaceholderId {
        Placeholder::id(self)
    }

    fn dtype(&self) -> &DType {
        Placeholder::dtype(self)
    }

    fn display_name(&self) -> &str {
        Placeholder::display_name(self)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

/// Erased placeholder handle stored in [`crate::expr::BoundExpr::Placeholder`].
#[derive(Clone)]
pub struct PlaceholderRef(Arc<dyn DynPlaceholder>);

impl PlaceholderRef {
    /// Erases a typed placeholder into a shared placeholder reference.
    pub fn new<P: Placeholder>(placeholder: P) -> Self {
        Self(Arc::new(placeholder))
    }

    /// Returns the placeholder id.
    pub fn id(&self) -> PlaceholderId {
        self.0.id()
    }

    /// Returns the placeholder dtype.
    pub fn dtype(&self) -> &DType {
        self.0.dtype()
    }

    /// Returns the placeholder SQL display name.
    pub fn display_name(&self) -> &str {
        self.0.display_name()
    }

    /// Returns whether this placeholder has the given concrete type.
    pub fn is<P: Placeholder>(&self) -> bool {
        self.0.as_any().is::<P>()
    }
}

impl Debug for PlaceholderRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlaceholderRef")
            .field("id", &self.id())
            .field("dtype", self.dtype())
            .field("payload", &DebugPlaceholder(self.0.as_ref()))
            .finish()
    }
}

struct DebugPlaceholder<'a>(&'a dyn DynPlaceholder);

impl Debug for DebugPlaceholder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_debug(f)
    }
}

impl Display for PlaceholderRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}()", self.display_name())
    }
}

impl PartialEq for PlaceholderRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id() && self.dtype() == other.dtype()
    }
}

impl Eq for PlaceholderRef {}

impl Hash for PlaceholderRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
        self.dtype().hash(state);
    }
}
