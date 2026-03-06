// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UUID extension type for Vortex.
//!
//! Provides a UUID extension type backed by `FixedSizeList(Primitive(U8), 16)` storage. Each UUID
//! is stored as 16 bytes in big-endian (network) byte order, matching [RFC 4122] and Arrow's
//! [canonical UUID extension].
//!
//! [RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122
//! [canonical UUID extension]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid

pub(crate) mod vtable;

/// The VTable for the UUID extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Uuid;
