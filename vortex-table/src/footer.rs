// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A data file's footer region: the bytes from its first footer segment to its end.
//!
//! The region holds the postscript and everything it points to (dtype, layout, statistics, and
//! the segment map), so a reader given the region, or its offset and length, opens the file
//! without searching for the footer.

use twox_hash::XxHash3_64;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::DeserializeStep;
use vortex_file::EOF_SIZE;
use vortex_file::Footer;
use vortex_io::VortexReadAt;
use vortex_session::VortexSession;

/// Where a file's footer region is, and its bytes.
pub struct FooterRegion {
    /// The region's first byte in the file.
    pub offset: u64,
    /// The region, through the end of the file.
    pub bytes: ByteBuffer,
    /// The parsed footer.
    pub footer: Footer,
}

/// The hash a manifest records for a footer region, and index sidecars bind to: XXH3-64 over
/// the region's bytes. A rewritten file, even one with the same path and size, hashes
/// differently.
pub fn footer_hash(region: &[u8]) -> u64 {
    XxHash3_64::oneshot(region)
}

/// Reads the exact footer region of a file of `size` bytes: its end-of-file marker, then its
/// postscript, then whatever else the postscript points to.
pub async fn read_footer_region(
    read: &dyn VortexReadAt,
    size: u64,
    session: &VortexSession,
) -> VortexResult<FooterRegion> {
    let eof_size = EOF_SIZE as u64;
    if size < eof_size {
        vortex_bail!("a file of {size} bytes has no end-of-file marker");
    }
    let eof = read_host(read, size - eof_size, EOF_SIZE).await?;
    let postscript = u64::from(u16::from_le_bytes([eof[2], eof[3]]));
    let tail = (postscript + eof_size).min(size);
    let mut deserializer = Footer::deserializer(
        read_host(read, size - tail, usize::try_from(tail)?).await?,
        session.clone(),
    )
    .with_size(size);
    let footer = loop {
        match deserializer.deserialize()? {
            DeserializeStep::NeedMoreData { offset, len } => {
                deserializer.prefix_data(read_host(read, offset, len).await?);
            }
            DeserializeStep::NeedFileSize => vortex_bail!("the file size was given"),
            DeserializeStep::Done(footer) => break footer,
        }
    };
    let bytes = deserializer.buffer().clone();
    Ok(FooterRegion {
        offset: size - bytes.len() as u64,
        bytes,
        footer,
    })
}

/// Parses a footer from its exact region, of a file of `size` bytes.
pub fn parse_footer(
    region: ByteBuffer,
    size: u64,
    session: &VortexSession,
) -> VortexResult<Footer> {
    let mut deserializer = Footer::deserializer(region, session.clone()).with_size(size);
    match deserializer.deserialize()? {
        DeserializeStep::Done(footer) => Ok(footer),
        DeserializeStep::NeedMoreData { offset, len } => {
            vortex_bail!("the footer region is short: it needs {len} more bytes at {offset}")
        }
        DeserializeStep::NeedFileSize => vortex_bail!("the file size was given"),
    }
}

async fn read_host(read: &dyn VortexReadAt, offset: u64, len: usize) -> VortexResult<ByteBuffer> {
    read.read_at(offset, len, Alignment::none())
        .await?
        .try_into_host()?
        .await
}
