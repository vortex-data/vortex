use std::any::Any;
use std::fmt::{Debug, Formatter};

use vortex_error::{vortex_bail, vortex_panic, VortexResult};
use vortex_mask::Mask;

use crate::encoding::EncodingId;
use crate::visitor::ArrayVisitor;
use crate::{Array, ArrayRef, Canonical};

/// An encoding of an array that we cannot interpret.
///
/// Vortex allows for pluggable encodings. This can lead to issues when one process produces a file
/// using a custom encoding, and then another process without knowledge of the encoding attempts
/// to read it.
///
/// `OpaqueEncoding` allows deserializing these arrays. Many common operations will fail, but it
/// allows deserialization and introspection in a type-erased manner on the children and metadata.
///
/// We hold the original 16-bit encoding ID for producing helpful error messages.
#[derive(Debug, Clone, Copy)]
pub struct OpaqueEncoding(pub u16);
