// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vector-search-bench` vector similarity-search benchmark over several datasets.

pub mod compression;
pub mod expression;
pub mod ingest;
pub mod prepare;

use std::sync::LazyLock;

use vortex::VortexSessionDefault;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    // SAFETY: called from inside the LazyLock initializer, before any other access to
    // `SESSION`. The first thread to dereference SESSION runs this once.
    unsafe { std::env::set_var(vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV, "1") };

    let session = VortexSession::default().with_tokio();
    vortex_tensor::initialize(&session);
    session
});
