// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrays as content-addressed blobs: serialized bytes plus what is needed to decode them.

use anyhow::bail;
use serde::Deserialize;
use serde::Serialize;
use vortex::buffer::ByteBuffer;
use vortex::session::VortexSession;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_session::registry::Id;
use vortex_session::registry::ReadContext;

use crate::key::bytes_digest;

/// How to decode a stored blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    /// The SHA-256 of the blob bytes; also its file name in the store.
    pub hash: String,
    /// The encoding ids the blob's indices refer to.
    pub ids: Vec<String>,
    /// The primitive type, e.g. `i64`.
    pub ptype: String,
    /// Whether the array is nullable.
    pub nullable: bool,
    /// The number of rows.
    pub len: u64,
    /// The blob size in bytes: buffers plus metadata, as written.
    pub bytes: u64,
}

/// Serializes a primitive-typed array. Returns its reference and its bytes.
pub fn serialize(array: &ArrayRef, session: &VortexSession) -> anyhow::Result<(BlobRef, Vec<u8>)> {
    let ctx = ArrayContext::empty();
    let buffers = array.serialize(&ctx, session, &SerializeOptions::default())?;
    let mut bytes = Vec::with_capacity(buffers.iter().map(|b| b.len()).sum());
    for buffer in &buffers {
        bytes.extend_from_slice(buffer.as_ref());
    }
    let dtype = array.dtype();
    let DType::Primitive(ptype, nullability) = dtype else {
        bail!("only primitive arrays are stored for now, got {dtype}");
    };
    Ok((
        BlobRef {
            hash: bytes_digest(&bytes),
            ids: ctx
                .to_ids()
                .iter()
                .map(|id| id.as_str().to_string())
                .collect(),
            ptype: ptype.to_string(),
            nullable: *nullability == Nullability::Nullable,
            len: array.len() as u64,
            bytes: bytes.len() as u64,
        },
        bytes,
    ))
}

/// Decodes a blob written by [`serialize`].
pub fn deserialize(
    blob: &BlobRef,
    bytes: Vec<u8>,
    session: &VortexSession,
) -> anyhow::Result<ArrayRef> {
    let ptype = parse_ptype(&blob.ptype)?;
    let nullability = if blob.nullable {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let ids: Vec<Id> = blob.ids.iter().map(|id| Id::from(id.as_str())).collect();
    let serialized = SerializedArray::try_from(ByteBuffer::from(bytes))?;
    Ok(serialized.decode(
        &DType::Primitive(ptype, nullability),
        usize::try_from(blob.len)?,
        &ReadContext::new(ids),
        session,
    )?)
}

/// The size of an array as written: its buffers plus the flatbuffer holding its metadata.
pub fn serialized_size(array: &ArrayRef, session: &VortexSession) -> anyhow::Result<u64> {
    let buffers = array.serialize(
        &ArrayContext::empty(),
        session,
        &SerializeOptions::default(),
    )?;
    Ok(buffers.iter().map(|b| b.len() as u64).sum())
}

fn parse_ptype(name: &str) -> anyhow::Result<PType> {
    Ok(match name {
        "u8" => PType::U8,
        "u16" => PType::U16,
        "u32" => PType::U32,
        "u64" => PType::U64,
        "i8" => PType::I8,
        "i16" => PType::I16,
        "i32" => PType::I32,
        "i64" => PType::I64,
        "f16" => PType::F16,
        "f32" => PType::F32,
        "f64" => PType::F64,
        other => bail!("unknown ptype `{other}`"),
    })
}
