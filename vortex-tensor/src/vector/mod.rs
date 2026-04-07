// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector extension type for fixed-length float vectors (e.g., embeddings).

/// The Vector extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector;

mod matcher;

pub use matcher::AnyVector;
pub use matcher::VectorMatcherMetadata;

mod vtable;
