// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Vortex crate containing core logic for encoding and memory representation of [arrays](ArrayRef).
//!
//! At the heart of Vortex are [arrays](ArrayRef).
//!
//! Arrays are typed views of memory buffers that hold [scalars](crate::scalar::Scalar). These
//! buffers can be held in a number of physical encodings to perform lightweight compression that
//! exploits the particular data distribution of the array's values.
//!
//! Every data type recognized by Vortex also has a canonical physical encoding format, which
//! arrays can be [canonicalized](Canonical) into for ease of access in compute functions.
//!
//! # Core Handles
//!
//! [`ArrayRef`] is the erased, shared handle used by most public APIs. It carries the logical
//! [`DType`], row count, encoding id, children, buffers, and statistics for an
//! array tree. Use it when an API should accept any encoding.
//!
//! [`Array<V>`] is the typed owned handle for a known encoding `V: VTable`. It wraps an
//! [`ArrayRef`] and dereferences to the encoding-specific `V::TypedArrayData`.
//!
//! [`ArrayView<V>`] is the lightweight typed borrow handed to vtable methods. It exposes both the
//! shared [`ArrayRef`] metadata and the encoding-specific data without cloning the handle.
//!
//! [`ArrayParts<V>`] is the construction boundary for typed arrays. It groups externally supplied
//! logical metadata and encoding data, then [`Array::try_from_parts`] validates that they agree.
//!
//! # Logical Types and Physical Encodings
//!
//! A [`DType`] describes the logical values an array may hold. It does not
//! describe the memory layout. For example, a `DType::Primitive(I32, Nullable)` can be stored as a
//! canonical [`PrimitiveArray`], a dictionary, a slice, or a
//! compressed external encoding.
//!
//! The [`Canonical`] enum names the default uncompressed encoding for each logical family. Execution
//! normally moves an array tree toward canonical form, but canonicalization is shallow: children of
//! canonical struct/list arrays may still be encoded.
//!
//! # Built-in, Lazy, and Experimental Arrays
//!
//! Built-in arrays live in [`arrays`]. Some are canonical (`PrimitiveArray`, `StructArray`,
//! `VarBinViewArray`); others are utility or lazy arrays such as [`ChunkedArray`],
//! [`ConstantArray`], [`FilterArray`], [`SliceArray`], and [`ScalarFnArray`].
//! Lazy arrays defer work so compute kernels can operate on encoded data or prune children
//! before materialization.
//!
//! Experimental arrays are public because they are used inside Vortex, but their storage contracts
//! may still move. Prefer the higher-level constructors and accessors documented on each array
//! module rather than relying on child slot order.
//!
//! # Nulls and Scalars
//!
//! [`Validity`](crate::validity::Validity) separates nullness from values. It can be a cheap
//! constant state (`NonNullable`, `AllValid`, `AllInvalid`) or a boolean array that may itself be
//! encoded. [`Scalar`](crate::scalar::Scalar) is the single-value counterpart: it pairs a
//! [`DType`] with an optional [`ScalarValue`](crate::scalar::ScalarValue).
//!
//! # Extending Vortex
//!
//! New array encodings implement [`VTable`], usually through the local `array_slots!` and
//! `vtable!` patterns used by built-ins. The important extension contracts are:
//!
//! - [`VTable::validate`] checks that externally supplied dtype, length, slots, and data agree.
//! - [`VTable::execute`] returns an [`ExecutionResult`] that makes progress toward canonical form.
//! - [`OperationsVTable`] provides scalar access.
//! - [`ValidityVTable`] exposes validity only for nullable arrays.
//!
//! New logical extension dtypes implement [`ExtVTable`](crate::dtype::extension::ExtVTable) and
//! store values in an ordinary Vortex storage dtype.
//!
//! [`PrimitiveArray`]: crate::arrays::PrimitiveArray
//! [`DType`]: crate::dtype::DType
//! [`ChunkedArray`]: crate::arrays::ChunkedArray
//! [`ConstantArray`]: crate::arrays::ConstantArray
//! [`FilterArray`]: crate::arrays::FilterArray
//! [`SliceArray`]: crate::arrays::SliceArray
//! [`ScalarFnArray`]: crate::arrays::ScalarFnArray

extern crate self as vortex_array;

use std::sync::LazyLock;

pub use array::*;
pub use canonical::*;
pub use columnar::*;
pub use executor::*;
pub use hash::*;
pub use mask_future::*;
pub use metadata::*;
pub use smallvec;
pub use vortex_array_macros::array_slots;
use vortex_session::VortexSession;
use vortex_session::registry::Context;

use crate::aggregate_fn::session::AggregateFnSession;
use crate::arrow::ArrowSession;
use crate::dtype::session::DTypeSession;
use crate::memory::MemorySession;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::kernels::KernelSession;
use crate::scalar_fn::session::ScalarFnSession;
use crate::session::ArraySession;
use crate::stats::session::StatsSession;

pub mod accessor;
pub mod aggregate_fn;
#[doc(hidden)]
pub mod aliases;
mod arc_swap_map;
mod array;
pub mod arrays;
pub mod arrow;
pub mod buffer;
pub mod builders;
pub mod builtins;
mod canonical;
mod columnar;
pub mod compute;
pub mod display;
pub mod dtype;
mod executor;
pub mod expr;
mod expression;
pub mod extension;
mod hash;
pub mod iter;
pub mod kernel;
pub mod mask;
mod mask_future;
pub mod matcher;
pub mod memory;
mod metadata;
pub mod normalize;
pub mod optimizer;
mod partial_ord;
pub mod patches;
pub mod scalar;
pub mod scalar_fn;
pub mod search_sorted;
pub mod serde;
pub mod session;
pub mod stats;
pub mod stream;
#[cfg(any(test, feature = "_test-harness"))]
pub mod test_harness;
pub mod validity;
pub mod variants;

pub mod flatbuffers {
    //! Re-exported autogenerated code from the core Vortex flatbuffer definitions.
    pub use vortex_flatbuffers::array::*;
}

/// Register vortex-array's built-in session-scoped kernels into the active
/// [`ArrayKernels`](crate::optimizer::kernels::ArrayKernels) registry.
///
/// If the session contains a [`KernelSession`], this registers into its registry. Sessions that use
/// [`KernelSession::default`] already receive these built-in kernels.
pub fn initialize(session: &VortexSession) {
    if session.kernels_opt().is_some() {
        arrays::initialize(session);
    }
}

/// Builds a fresh [`VortexSession`] registered with all of vortex-array's built-in session
/// variables: arrays, dtypes, scalar functions, stats, optimizer kernels, aggregate functions,
/// Arrow conversion, and memory.
///
/// Each call returns an independent session (with its own registries), so callers may register
/// additional encodings or kernels into it without affecting any other session. This does not
/// register file, layout, or runtime state — those live in higher-level crates.
pub fn array_session() -> VortexSession {
    VortexSession::builder()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<DTypeSession>()
        .with::<ScalarFnSession>()
        .with::<StatsSession>()
        .with::<AggregateFnSession>()
        .with::<ArrowSession>()
        .with::<MemorySession>()
        .build()
}

// TODO(ngates): canonicalize doesn't currently take a session, therefore we cannot invoke execute
//  from the new array encodings to support back-compat for legacy encodings. So we hold a session
//  here...
pub static LEGACY_SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

pub type ArrayContext = Context<ArrayPluginRef>;
