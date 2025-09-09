// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use flatbuffers::root;
use vortex::buffer::{Alignment, ByteBuffer};
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::file::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};
use vortex::flatbuffers::footer as fb;

#[derive(Debug, clap::Parser)]
pub struct InspectArgs {
    /// What to inspect
    #[clap(subcommand)]
    pub mode: Option<InspectMode>,

    /// Path to the Vortex file to inspect
    pub file: PathBuf,
}

#[derive(Debug, clap::Subcommand)]
pub enum InspectMode {
    /// Read and display the EOF marker (8 bytes at end of file)
    Eof,

    /// Read and display the postscript
    Postscript,

    /// Read and display all footer segments
    Footer,
}

pub fn exec_inspect(args: InspectArgs) -> anyhow::Result<()> {
    let mut inspector = VortexInspector::new(args.file.clone())?;

    println!("File: {}", args.file.display());
    println!("Size: {} bytes", inspector.file_size);
    println!();

    let mode = args.mode.unwrap_or(InspectMode::Footer);

    match mode {
        InspectMode::Eof => {
            let eof = inspector.read_eof()?;
            eof.display();
        }
        InspectMode::Postscript => {
            let eof = inspector.read_eof()?;
            eof.display();

            if !eof.valid_magic {
                anyhow::bail!("Invalid magic bytes, cannot read postscript");
            }

            let postscript = inspector.read_postscript(eof.postscript_size)?;
            postscript.display();
        }
        InspectMode::Footer => {
            let eof = match inspector.read_eof() {
                Ok(eof) => {
                    eof.display();
                    eof
                }
                Err(e) => {
                    eprintln!("Error reading EOF: {}", e);
                    return Err(e.into());
                }
            };

            if !eof.valid_magic {
                eprintln!("\nError: Invalid magic bytes, stopping here");
                return Ok(());
            }

            let postscript = match inspector.read_postscript(eof.postscript_size) {
                Ok(ps) => {
                    ps.display();
                    ps
                }
                Err(e) => {
                    eprintln!("\nError reading postscript: {}", e);
                    return Ok(());
                }
            };

            match inspector.read_footer_segments(&postscript) {
                Ok(footer) => footer.display(),
                Err(e) => {
                    eprintln!("\nError reading footer segments: {}", e);
                }
            }
        }
    }

    Ok(())
}

struct VortexInspector {
    file: File,
    file_size: u64,
}

impl VortexInspector {
    fn new(path: PathBuf) -> VortexResult<Self> {
        let mut file =
            File::open(&path).map_err(|e| vortex_err!("Failed to open file {:?}: {}", path, e))?;

        let file_size = file
            .seek(SeekFrom::End(0))
            .map_err(|e| vortex_err!("Failed to get file size: {}", e))?;

        Ok(Self { file, file_size })
    }

    fn read_eof(&mut self) -> VortexResult<EofInfo> {
        if self.file_size < EOF_SIZE as u64 {
            vortex_bail!(
                "File too small ({} bytes) to contain EOF marker (requires {} bytes)",
                self.file_size,
                EOF_SIZE
            );
        }

        let mut eof_bytes = [0u8; EOF_SIZE];
        self.file
            .seek(SeekFrom::End(-(EOF_SIZE as i64)))
            .map_err(|e| vortex_err!("Failed to seek to EOF: {}", e))?;
        self.file
            .read_exact(&mut eof_bytes)
            .map_err(|e| vortex_err!("Failed to read EOF bytes: {}", e))?;

        let version = u16::from_le_bytes([eof_bytes[0], eof_bytes[1]]);
        let postscript_size = u16::from_le_bytes([eof_bytes[2], eof_bytes[3]]);
        let magic_bytes = [eof_bytes[4], eof_bytes[5], eof_bytes[6], eof_bytes[7]];

        Ok(EofInfo {
            version,
            postscript_size,
            magic_bytes,
            valid_magic: magic_bytes == MAGIC_BYTES,
        })
    }

    fn read_postscript(&mut self, postscript_size: u16) -> VortexResult<PostscriptInfo> {
        let postscript_offset = self.file_size - EOF_SIZE as u64 - postscript_size as u64;

        let mut postscript_bytes = vec![0u8; postscript_size as usize];
        self.file
            .seek(SeekFrom::Start(postscript_offset))
            .map_err(|e| vortex_err!("Failed to seek to postscript: {}", e))?;
        self.file
            .read_exact(&mut postscript_bytes)
            .map_err(|e| vortex_err!("Failed to read postscript: {}", e))?;

        let postscript_buffer = ByteBuffer::from(postscript_bytes);
        let fb_postscript = root::<fb::Postscript>(&postscript_buffer)
            .map_err(|e| vortex_err!("Failed to parse postscript flatbuffer: {}", e))?;

        let dtype = fb_postscript.dtype().map(|s| SegmentInfo {
            offset: s.offset(),
            length: s.length(),
            alignment: Alignment::from_exponent(s.alignment_exponent()),
        });

        let layout = fb_postscript
            .layout()
            .map(|s| SegmentInfo {
                offset: s.offset(),
                length: s.length(),
                alignment: Alignment::from_exponent(s.alignment_exponent()),
            })
            .ok_or_else(|| vortex_err!("Postscript missing layout segment"))?;

        let statistics = fb_postscript.statistics().map(|s| SegmentInfo {
            offset: s.offset(),
            length: s.length(),
            alignment: Alignment::from_exponent(s.alignment_exponent()),
        });

        let footer = fb_postscript
            .footer()
            .map(|s| SegmentInfo {
                offset: s.offset(),
                length: s.length(),
                alignment: Alignment::from_exponent(s.alignment_exponent()),
            })
            .ok_or_else(|| vortex_err!("Postscript missing footer segment"))?;

        Ok(PostscriptInfo {
            dtype,
            layout,
            statistics,
            footer,
        })
    }

    fn read_footer_segments(
        &mut self,
        postscript: &PostscriptInfo,
    ) -> VortexResult<FooterSegments> {
        // Read footer segment

        let mut footer_bytes = vec![0u8; postscript.footer.length as usize];
        self.file
            .seek(SeekFrom::Start(postscript.footer.offset))
            .map_err(|e| vortex_err!("Failed to seek to footer: {}", e))?;
        self.file
            .read_exact(&mut footer_bytes)
            .map_err(|e| vortex_err!("Failed to read footer: {}", e))?;

        let footer_buffer = ByteBuffer::from(footer_bytes);
        let fb_footer = root::<fb::Footer>(&footer_buffer)
            .map_err(|e| vortex_err!("Failed to parse footer flatbuffer: {}", e))?;

        let segment_count = fb_footer
            .segment_specs()
            .map(|segs| segs.len())
            .unwrap_or(0);

        let mut segments = Vec::new();
        if let Some(fb_segments) = fb_footer.segment_specs() {
            for seg in fb_segments {
                segments.push(SegmentInfo {
                    offset: seg.offset(),
                    length: seg.length(),
                    alignment: Alignment::from_exponent(seg.alignment_exponent()),
                });
            }
        }

        Ok(FooterSegments {
            segment_count,
            total_data_size: segments.iter().map(|s| s.length as u64).sum(),
            segments,
        })
    }
}

#[derive(Debug)]
struct EofInfo {
    version: u16,
    postscript_size: u16,
    magic_bytes: [u8; 4],
    valid_magic: bool,
}

#[derive(Debug, Clone)]
struct SegmentInfo {
    offset: u64,
    length: u32,
    alignment: Alignment,
}

#[derive(Debug)]
struct PostscriptInfo {
    pub dtype: Option<SegmentInfo>,
    pub layout: SegmentInfo,
    pub statistics: Option<SegmentInfo>,
    pub footer: SegmentInfo,
}

#[derive(Debug)]
struct FooterSegments {
    segment_count: usize,
    segments: Vec<SegmentInfo>,
    total_data_size: u64,
}

impl EofInfo {
    fn display(&self) {
        println!("=== EOF Marker ===");
        println!("Version: {} (current: {})", self.version, VERSION);
        println!("Postscript size: {} bytes", self.postscript_size);
        println!(
            "Magic bytes: {} ({})",
            std::str::from_utf8(&self.magic_bytes).unwrap_or("<invalid utf8>"),
            if self.valid_magic { "VALID" } else { "INVALID" }
        );

        if self.postscript_size > MAX_FOOTER_SIZE {
            println!(
                "WARNING: Postscript size exceeds maximum ({} > {})",
                self.postscript_size, MAX_FOOTER_SIZE
            );
        }
    }
}

impl SegmentInfo {
    fn display(&self, name: &str) {
        println!(
            "  {}: offset={}, length={}, alignment={}",
            name, self.offset, self.length, self.alignment
        );
    }
}

impl PostscriptInfo {
    fn display(&self) {
        println!("\n=== Postscript ===");
        if let Some(ref dtype) = self.dtype {
            dtype.display("DType");
        } else {
            println!("  DType: <not embedded>");
        }
        self.layout.display("Layout");
        if let Some(ref stats) = self.statistics {
            stats.display("Statistics");
        } else {
            println!("  Statistics: <not present>");
        }
        self.footer.display("Footer");
    }
}

impl FooterSegments {
    fn display(&self) {
        println!("\n=== Footer Segments ===");
        println!("Total segments: {}", self.segment_count);
        println!("Total data size: {} bytes", self.total_data_size);

        if !self.segments.is_empty() {
            println!("\nSegment details:");
            for (i, segment) in self.segments.iter().enumerate() {
                println!(
                    "  [{}] offset={}, length={}, alignment={}",
                    i, segment.offset, segment.length, segment.alignment
                );
            }
        }
    }
}
