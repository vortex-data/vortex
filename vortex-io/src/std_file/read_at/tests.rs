// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "tokio")]
#![expect(clippy::cast_possible_truncation, reason = "fixture sizes are small")]

use std::io::Write;

use rstest::rstest;
use tempfile::NamedTempFile;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::runtime::tokio::TokioRuntime;
use crate::std_file::FileReadAt;
use crate::std_file::FileReadAtOptions;

/// A file length that is deliberately not a multiple of any plausible block size, so the final
/// block is partial and every direct read near the end returns short.
const FILE_LEN: usize = 40_000 + 37;

fn contents() -> Vec<u8> {
    (0..FILE_LEN).map(|i| (i % 251) as u8).collect()
}

fn temp_file(contents: &[u8]) -> VortexResult<NamedTempFile> {
    let mut file = NamedTempFile::new()?;
    file.write_all(contents)?;
    file.flush()?;
    Ok(file)
}

fn direct_options() -> FileReadAtOptions {
    #[cfg(target_os = "linux")]
    {
        FileReadAtOptions::default().with_direct_io()
    }
    #[cfg(not(target_os = "linux"))]
    {
        FileReadAtOptions::default()
    }
}

fn open(path: &std::path::Path, options: FileReadAtOptions) -> VortexResult<FileReadAt> {
    FileReadAt::open_with_options(
        path,
        TokioRuntime::current(),
        BufferAllocatorRef::statically_allocated(),
        options,
    )
}

#[test]
fn options_default_to_buffered_io() {
    assert!(!FileReadAtOptions::default().direct_io());
}

#[cfg(target_os = "linux")]
#[test]
fn options_enable_direct_io() {
    assert!(FileReadAtOptions::default().with_direct_io().direct_io());
}

/// Direct reads must return exactly the requested window, including when the request straddles or
/// sits inside a block, and when it runs up against a partial final block.
#[rstest]
#[case(0, 1)]
#[case(0, FILE_LEN)]
#[case(1, 4095)]
#[case(4095, 2)]
#[case(4096, 4096)]
#[case(511, 8193)]
#[case(FILE_LEN as u64 - 1, 1)]
#[case(FILE_LEN as u64 - 4097, 4097)]
#[case(35_000, FILE_LEN - 35_000)]
#[tokio::test]
async fn direct_reads_return_the_requested_bytes(
    #[case] offset: u64,
    #[case] length: usize,
) -> VortexResult<()> {
    let expected = contents();
    let file = temp_file(&expected)?;

    let direct = open(file.path(), direct_options())?;
    let buffered = open(file.path(), FileReadAtOptions::default())?;

    let direct = direct.read_at(offset, length, Alignment::none()).await?;
    let buffered = buffered.read_at(offset, length, Alignment::none()).await?;

    let window = &expected[offset as usize..offset as usize + length];
    assert_eq!(direct.to_host().await.as_slice(), window);
    assert_eq!(buffered.to_host().await.as_slice(), window);
    Ok(())
}

/// Widening a read to block boundaries must not cost the caller their alignment: a segment stored
/// at a naturally aligned file offset is still naturally aligned once sliced out of the block.
#[rstest]
#[case(8, Alignment::new(8))]
#[case(4104, Alignment::new(8))]
#[case(256, Alignment::new(256))]
#[case(8192, Alignment::new(4096))]
#[tokio::test]
async fn direct_reads_preserve_requested_alignment(
    #[case] offset: u64,
    #[case] alignment: Alignment,
) -> VortexResult<()> {
    let expected = contents();
    let file = temp_file(&expected)?;
    let reader = open(file.path(), direct_options())?;

    let length = 1024;
    let buffer = reader.read_at(offset, length, alignment).await?;
    let host = buffer.to_host().await;

    assert_eq!(host.alignment(), alignment);
    assert!(host.is_aligned(alignment));
    assert_eq!(
        host.as_slice(),
        &expected[offset as usize..offset as usize + length]
    );
    Ok(())
}

#[tokio::test]
async fn direct_reads_past_the_end_of_the_file_fail() -> VortexResult<()> {
    let file = temp_file(&contents())?;
    let reader = open(file.path(), direct_options())?;

    assert!(
        reader
            .read_at(FILE_LEN as u64 - 8, 4096, Alignment::none())
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn direct_reads_of_zero_length_are_empty() -> VortexResult<()> {
    let file = temp_file(&contents())?;
    let reader = open(file.path(), direct_options())?;

    let buffer = reader.read_at(37, 0, Alignment::new(8)).await?;
    assert_eq!(buffer.len(), 0);
    Ok(())
}

/// Direct I/O is unavailable on some filesystems (tmpfs, overlayfs) and on non-Linux platforms.
/// Opening must still succeed there, silently serving buffered reads.
#[tokio::test]
async fn opening_never_fails_when_direct_io_is_unavailable() -> VortexResult<()> {
    let expected = contents();
    let file = temp_file(&expected)?;
    let reader = open(file.path(), direct_options())?;

    let buffer = reader.read_at(0, 128, Alignment::none()).await?;
    assert_eq!(buffer.to_host().await.as_slice(), &expected[..128]);
    Ok(())
}

/// Wherever the platform can actually serve `O_DIRECT`, requesting it must take effect rather than
/// quietly degrading, otherwise the option would be untestable and unmeasurable.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn direct_io_is_used_when_the_filesystem_supports_it() -> VortexResult<()> {
    let file = temp_file(&contents())?;
    if crate::std_file::open_direct(file.path()).is_err() {
        return Ok(());
    }

    assert!(open(file.path(), direct_options())?.is_direct());
    assert!(!open(file.path(), FileReadAtOptions::default())?.is_direct());
    Ok(())
}
