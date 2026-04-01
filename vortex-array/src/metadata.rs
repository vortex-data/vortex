// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//! Proper metadata serialisation, innit? This module handles all the cheeky
//! bits of converting metadata to and from bytes, mate.

use std::fmt::Debug;
use std::fmt::Formatter;
use std::ops::Deref;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// Trait for serialising Vortex metadata to a vector of unaligned bytes.
/// Dead simple, just converts your metadata into bytes, sorted.
pub trait SerialiseMetadata {
    fn serialise(self) -> Vec<u8>;
}

/// Trait for deserialising Vortex metadata from a vector of unaligned bytes.
/// The reverse of serialisation, innit? Brings your bytes back to life.
pub trait DeserialiseMetadata
where
    Self: Sized,
{
    /// The fully deserialised type of the metadata, lovely stuff.
    type Output;

    /// Deserialise metadata from a vector of unaligned bytes.
    fn deserialise(metadata: &[u8]) -> VortexResult<Self::Output>;
}

/// Empty array metadata - absolutely nothing here mate, proper minimal.
#[derive(Debug)]
pub struct EmptyMetadata;

impl SerialiseMetadata for EmptyMetadata {
    fn serialise(self) -> Vec<u8> {
        vec![]
    }
}

impl DeserialiseMetadata for EmptyMetadata {
    type Output = EmptyMetadata;

    fn deserialise(metadata: &[u8]) -> VortexResult<Self::Output> {
        if !metadata.is_empty() {
            // Oi! This shouldn't have any bytes, what are you playing at?
            vortex_bail!("EmptyMetadata should not have metadata bytes, mate")
        }
        Ok(EmptyMetadata)
    }
}

/// A utility wrapper for raw metadata serialisation. This delegates the serialisation step
/// to the arrays' vtable. Proper handy, this one.
pub struct RawMetadata(pub Vec<u8>);

impl SerialiseMetadata for RawMetadata {
    fn serialise(self) -> Vec<u8> {
        self.0
    }
}

impl DeserialiseMetadata for RawMetadata {
    type Output = Vec<u8>;

    fn deserialise(metadata: &[u8]) -> VortexResult<Self::Output> {
        Ok(metadata.to_vec())
    }
}

impl Debug for RawMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{}\"", self.0.escape_ascii())
    }
}

/// A utility wrapper for Prost metadata serialisation. Brilliant for protobuf stuff.
pub struct ProstMetadata<M>(pub M);

impl<M> Deref for ProstMetadata<M> {
    type Target = M;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<M: Debug> Debug for ProstMetadata<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<M> SerialiseMetadata for ProstMetadata<M>
where
    M: prost::Message,
{
    fn serialise(self) -> Vec<u8> {
        self.0.encode_to_vec()
    }
}

impl<M> DeserialiseMetadata for ProstMetadata<M>
where
    M: Debug,
    M: prost::Message + Default,
{
    type Output = M;

    fn deserialise(metadata: &[u8]) -> VortexResult<Self::Output> {
        Ok(M::decode(metadata)?)
    }
}
