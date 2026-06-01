// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vector-search-bench` vector similarity-search benchmark over several datasets.

pub mod compression;
pub mod display;
pub mod distortion;
pub mod expression;
pub mod ingest;
pub mod prepare;
pub mod query;
pub mod scan;

use std::sync::LazyLock;

use anyhow::Result;
use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex_bench::vector_dataset::TrainLayout;
use vortex_bench::vector_dataset::VectorDataset;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    // SAFETY: called from inside the LazyLock initializer, before any other access to
    // `SESSION`. The first thread to dereference SESSION runs this once.
    unsafe { std::env::set_var(vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV, "1") };

    let session = VortexSession::default().with_tokio();
    vortex_tensor::initialize(&session);
    session
});

/// Resolve a dataset's [`TrainLayout`].
///
/// Every benchmark has different sets of possible dataset layouts available. The user **must**
/// provide one if there are multiple layouts. But if a dataset only has 1 layout, we can choose
/// that for them as the default.
pub fn resolve_layout(
    dataset: VectorDataset,
    requested: Option<TrainLayout>,
) -> Result<TrainLayout> {
    let layouts = dataset.layouts();

    match requested {
        Some(layout) => {
            dataset.validate_layout(layout)?;
            Ok(layout)
        }
        None => {
            if layouts.len() == 1 {
                Ok(layouts[0].layout())
            } else {
                let allowed = layouts
                    .iter()
                    .map(|s| s.layout().label())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!(
                    "dataset {} hosts multiple layouts ([{}]): pass --layout to pick one",
                    dataset.name(),
                    allowed,
                );
            }
        }
    }
}
