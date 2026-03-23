// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use crate::v2::layout::LayoutId;

/// The vtable for a pluggable layout.
pub trait LayoutVTable: 'static + Sized + Clone + Send + Sync {
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    fn id(&self) -> LayoutId;
}
