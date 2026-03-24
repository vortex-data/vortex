// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use flatbuffers::Follow;
use parking_lot::RwLock;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::layout as fbl;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentSource;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::session::LayoutSessionExt;

/// A potentially un-resolved layout child.
///
/// Internally, this object lazily caches deserialization of layout flatbuffers.
pub struct LayoutChild(Arc<RwLock<Inner>>);

enum Inner {
    Owned(LayoutRef),
    Viewed {
        fb: FlatBuffer,
        loc: usize,
        ids: Arc<[LayoutId]>,
        context: ReadContext,
        source: Arc<dyn SegmentSource>,
        session: VortexSession,
    },
}

impl LayoutChild {
    /// Resolve the layout child by passing the child's expected DType.
    pub fn resolve(&self, dtype: &DType) -> VortexResult<LayoutRef> {
        if let Inner::Owned(owned) = &*self.0.read() {
            debug_assert!(
                owned.dtype() == dtype,
                "In-memory layout child resolved with incorrect DType"
            );
            return Ok(owned.clone());
        }

        let mut guard = self.0.write();
        Ok(match &mut *guard {
            Inner::Owned(owned) => owned.clone(),
            Inner::Viewed {
                fb,
                loc,
                ids,
                context,
                source,
                session,
            } => {
                let fb_layout = unsafe { fbl::Layout::follow(fb.as_slice(), *loc) };
                let id = ids.get(fb_layout.encoding() as usize).ok_or_else(|| {
                    vortex_err!("Interned layout ID out of bounds: {}", fb_layout.encoding())
                })?;

                let plugin = session
                    .layouts2()
                    .registry()
                    .find(id)
                    .ok_or_else(|| vortex_err!("Layout {} not found in registry", id))?;

                let metadata = fb_layout
                    .metadata()
                    .map(|bytes| bytes.bytes())
                    .unwrap_or(&[]);

                let children = fb_layout
                    .children()
                    .map(|children| {
                        children
                            .iter()
                            .map(|child| {
                                LayoutChild(Arc::new(RwLock::new(Inner::Viewed {
                                    fb: fb.clone(),
                                    loc: child._tab.loc(),
                                    ids: ids.clone(),
                                    context: context.clone(),
                                    source: source.clone(),
                                    session: session.clone(),
                                })))
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let layout = plugin.deserialize(dtype, metadata, children, source, session)?;

                // Update the layout child to cache the owned layout
                *guard = Inner::Owned(layout.clone());

                layout
            }
        })
    }
}
