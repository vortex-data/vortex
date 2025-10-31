// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::layouts::chunked::ChunkedLayoutEncoding;
use crate::layouts::dict::DictLayoutEncoding;
use crate::layouts::flat::FlatLayoutEncoding;
use crate::layouts::struct_::StructLayoutEncoding;
use crate::layouts::zoned::ZonedLayoutEncoding;
use crate::LayoutEncodingRef;
use std::ops::Deref;
use vortex_session::registry::Registry;
use vortex_session::SessionExt;

pub type LayoutRegistry = Registry<LayoutEncodingRef>;

/// Session state for layout encodings.
#[derive(Debug)]
pub struct LayoutSession {
    layouts: LayoutRegistry,
}

impl Default for LayoutSession {
    fn default() -> Self {
        let mut layouts = LayoutRegistry::default();

        // Register the built-in layout encodings.
        layouts.register_many([
            LayoutEncodingRef::new_ref(ChunkedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(FlatLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(StructLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(ZonedLayoutEncoding.as_ref()),
            LayoutEncodingRef::new_ref(DictLayoutEncoding.as_ref()),
        ]);

        Self { layouts }
    }
}

/// Extension trait for accessing layout session data.
pub trait LayoutSessionExt: SessionExt {
    /// Register a layout encoding in the session, replacing any existing encoding with the same ID.
    fn register_layout(&self, layout: LayoutEncodingRef) {
        self.register_layouts([layout])
    }

    /// Register layout encodings in the session, replacing any existing encodings with the same IDs.
    fn register_layouts(&self, layouts: impl IntoIterator<Item = LayoutEncodingRef>) {
        self.get::<LayoutSession>().layouts.register_many(layouts);
    }

    /// Returns the layout encoding registry.
    fn layouts(&self) -> impl Deref<Target = LayoutRegistry> {
        self.get::<LayoutSession>().map(|v| &v.layouts)
    }
}
impl<S: SessionExt> LayoutSessionExt for S {}
