// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;

use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;

pub const DEFAULT_RDMA_PORT: u16 = 9900;

pub const OP_LIST: u8 = 1;
pub const OP_SIZE: u8 = 2;
pub const OP_READ: u8 = 3;
pub const OP_IPC_HANDLE: u8 = 4;

pub const STATUS_OK: u8 = 0;
pub const STATUS_ERR: u8 = 1;

pub async fn write_string<W: AsyncWrite + Unpin>(writer: &mut W, value: &str) -> io::Result<()> {
    let len = u32::try_from(value.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "string too long"))?;
    writer.write_u32_le(len).await?;
    writer.write_all(value.as_bytes()).await
}

pub async fn read_string<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<String> {
    let len = reader.read_u32_le().await?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf).await?;
    String::from_utf8(buf).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid utf-8"))
}

pub async fn write_error<W: AsyncWrite + Unpin>(writer: &mut W, msg: &str) -> io::Result<()> {
    writer.write_u8(STATUS_ERR).await?;
    write_string(writer, msg).await
}

pub async fn read_status<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<()> {
    match reader.read_u8().await? {
        STATUS_OK => Ok(()),
        STATUS_ERR => {
            let msg = read_string(reader).await?;
            Err(io::Error::other(msg))
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown status byte: {other}"),
        )),
    }
}
