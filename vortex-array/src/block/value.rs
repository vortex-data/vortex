// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The types of values a block can hold.

use std::fmt::Display;
use std::fmt::Formatter;

use crate::dtype::PType;

/// The type of the values a block holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BlockType {
    /// Values of a primitive type.
    Primitive(PType),
}

impl BlockType {
    /// The size of one value, in bytes.
    pub fn byte_width(self) -> usize {
        match self {
            BlockType::Primitive(ptype) => ptype.byte_width(),
        }
    }
}

impl From<PType> for BlockType {
    fn from(ptype: PType) -> Self {
        BlockType::Primitive(ptype)
    }
}

impl Display for BlockType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockType::Primitive(ptype) => ptype.fmt(f),
        }
    }
}

/// A type of value a block can hold: fixed-width, with no padding, every bit pattern of which is
/// a value.
pub trait BlockValue: Copy + Send + Sync + 'static + sealed::Sealed {
    /// The block type of this value.
    const BLOCK_TYPE: BlockType;
}

pub(crate) mod sealed {
    /// Prevents implementations of [`BlockValue`](super::BlockValue) outside this crate.
    pub trait Sealed {}
}
