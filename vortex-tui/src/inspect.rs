// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inspect Vortex file metadata and structure.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use flatbuffers::root;
use itertools::Itertools;
use serde::Serialize;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::EOF_SIZE;
use vortex::file::Footer;
use vortex::file::MAGIC_BYTES;
use vortex::file::MAX_POSTSCRIPT_SIZE;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VERSION;
use vortex::flatbuffers::footer as fb;
use vortex::layout::LayoutRef;
use vortex::session::VortexSession;

/// Command-line arguments for the inspect command.
#[derive(Debug, clap::Parser)]
pub struct InspectArgs {
    /// What to inspect.
    #[clap(subcommand)]
    pub mode: Option<InspectMode>,

    /// Path to the Vortex file to inspect.
    pub file: PathBuf,

    /// Output as JSON
    #[arg(long, global = true)]
    pub json: bool,
}

/// What component of the Vortex file to inspect.
#[derive(Debug, clap::Subcommand)]
pub enum InspectMode {
    /// Read and display the EOF marker (8 bytes at end of file).
    Eof,

    /// Read and display the postscript
    Postscript,

    /// Read and display all footer segments
    Footer,
}

/// JSON output structure for inspect command.
#[derive(Serialize)]
pub struct InspectOutput {
    /// Path to the inspected file.
    pub file_path: String,
    /// Size of the file in bytes.
    pub file_size: u64,
    /// EOF marker information.
    pub eof: EofInfoJson,
    /// Postscript information (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postscript: Option<PostscriptInfoJson>,
    /// Footer information (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub footer: Option<FooterInfoJson>,
}

/// EOF marker information for JSON output.
#[derive(Serialize)]
pub struct EofInfoJson {
    /// File format version.
    pub version: u16,
    /// Current supported version.
    pub current_version: u16,
    /// Postscript size in bytes.
    pub postscript_size: u16,
    /// Magic bytes as string.
    pub magic_bytes: String,
    /// Whether magic bytes are valid.
    pub valid_magic: bool,
}

/// Segment information for JSON output.
#[derive(Serialize)]
pub struct SegmentInfoJson {
    /// Offset in file.
    pub offset: u64,
    /// Length in bytes.
    pub length: u32,
    /// Alignment requirement.
    pub alignment: usize,
}

/// Postscript information for JSON output.
#[derive(Serialize)]
pub struct PostscriptInfoJson {
    /// DType segment info.
    pub dtype: Option<SegmentInfoJson>,
    /// Layout segment info.
    pub layout: SegmentInfoJson,
    /// Statistics segment info.
    pub statistics: Option<SegmentInfoJson>,
    /// Footer segment info.
    pub footer: SegmentInfoJson,
}

/// Footer information for JSON output.
#[derive(Serialize)]
pub struct FooterInfoJson {
    /// Total number of segments.
    pub total_segments: usize,
    /// Total data size in bytes.
    pub total_data_size: u64,
    /// Individual segment details.
    pub segments: Vec<FooterSegmentJson>,
}

/// Footer segment information for JSON output.
#[derive(Serialize)]
pub struct FooterSegmentJson {
    /// Segment index.
    pub index: usize,
    /// Start offset in file.
    pub offset: u64,
    /// End offset in file.
    pub end_offset: u64,
    /// Length in bytes.
    pub length: u32,
    /// Alignment requirement.
    pub alignment: usize,
    /// Path in layout tree.
    pub path: Option<String>,
}

/// Inspect Vortex file footer and metadata.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or its metadata cannot be read.
pub async fn exec_inspect(session: &VortexSession, args: InspectArgs) -> anyhow::Result<()> {
    let mut inspector = VortexInspector::new(session, args.file.clone())?;

    let mode = args.mode.unwrap_or(InspectMode::Footer);

    if args.json {
        exec_inspect_json(&mut inspector, &args.file, mode).await
    } else {
        exec_inspect_text(&mut inspector, &args.file, mode).await
    }
}

async fn exec_inspect_json(
    inspector: &mut VortexInspector<'_>,
    file_path: &Path,
    mode: InspectMode,
) -> anyhow::Result<()> {
    let eof = inspector.read_eof()?;
    let eof_json = EofInfoJson {
        version: eof.version,
        current_version: VERSION,
        postscript_size: eof.postscript_size,
        magic_bytes: std::str::from_utf8(&eof.magic_bytes)
            .unwrap_or("<invalid utf8>")
            .to_string(),
        valid_magic: eof.valid_magic,
    };

    let postscript_json =
        if matches!(mode, InspectMode::Postscript | InspectMode::Footer) && eof.valid_magic {
            inspector
                .read_postscript(eof.postscript_size)
                .ok()
                .map(|ps| PostscriptInfoJson {
                    dtype: ps.dtype.map(|s| SegmentInfoJson {
                        offset: s.offset,
                        length: s.length,
                        alignment: *s.alignment,
                    }),
                    layout: SegmentInfoJson {
                        offset: ps.layout.offset,
                        length: ps.layout.length,
                        alignment: *ps.layout.alignment,
                    },
                    statistics: ps.statistics.map(|s| SegmentInfoJson {
                        offset: s.offset,
                        length: s.length,
                        alignment: *s.alignment,
                    }),
                    footer: SegmentInfoJson {
                        offset: ps.footer.offset,
                        length: ps.footer.length,
                        alignment: *ps.footer.alignment,
                    },
                })
        } else {
            None
        };

    let footer_json =
        if matches!(mode, InspectMode::Footer) && eof.valid_magic && postscript_json.is_some() {
            inspector.read_footer().await.ok().map(|footer| {
                let segment_map = Arc::clone(footer.segment_map());
                let root_layout = Arc::clone(footer.layout());

                let mut segment_paths: Vec<Option<Vec<Arc<str>>>> = vec![None; segment_map.len()];
                let mut queue =
                    VecDeque::<(Vec<Arc<str>>, LayoutRef)>::from_iter([(Vec::new(), root_layout)]);
                while !queue.is_empty() {
                    let (path, layout) = queue.pop_front().vortex_expect("queue is not empty");
                    for segment in layout.segment_ids() {
                        segment_paths[*segment as usize] = Some(path.clone());
                    }
                    if let Ok(children) = layout.children() {
                        for (child_layout, child_name) in
                            children.into_iter().zip(layout.child_names())
                        {
                            let child_path = path.iter().cloned().chain([child_name]).collect();
                            queue.push_back((child_path, child_layout));
                        }
                    }
                }

                let segments: Vec<FooterSegmentJson> = segment_map
                    .iter()
                    .enumerate()
                    .map(|(i, segment)| FooterSegmentJson {
                        index: i,
                        offset: segment.offset,
                        end_offset: segment.offset + segment.length as u64,
                        length: segment.length,
                        alignment: *segment.alignment,
                        path: segment_paths[i]
                            .as_ref()
                            .map(|p| p.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(".")),
                    })
                    .collect();

                FooterInfoJson {
                    total_segments: segment_map.len(),
                    total_data_size: segment_map.iter().map(|s| s.length as u64).sum(),
                    segments,
                }
            })
        } else {
            None
        };

    let output = InspectOutput {
        file_path: file_path.display().to_string(),
        file_size: inspector.file_size,
        eof: eof_json,
        postscript: postscript_json,
        footer: footer_json,
    };

    let json_output = serde_json::to_string_pretty(&output)?;
    println!("{json_output}");

    Ok(())
}

async fn exec_inspect_text(
    inspector: &mut VortexInspector<'_>,
    file_path: &Path,
    mode: InspectMode,
) -> anyhow::Result<()> {
    println!("File: {}", file_path.display());
    println!("Size: {} bytes", inspector.file_size);
    println!();

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

            match inspector.read_postscript(eof.postscript_size) {
                Ok(ps) => {
                    ps.display();
                }
                Err(e) => {
                    eprintln!("\nError reading postscript: {}", e);
                    return Ok(());
                }
            };

            match inspector.read_footer().await {
                Ok(footer) => FooterSegments(footer).display(),
                Err(e) => {
                    eprintln!("\nError reading footer segments: {}", e);
                }
            }
        }
    }

    Ok(())
}

struct VortexInspector<'a> {
    session: &'a VortexSession,
    path: PathBuf,
    file: File,
    file_size: u64,
}

impl<'a> VortexInspector<'a> {
    fn new(session: &'a VortexSession, path: PathBuf) -> VortexResult<Self> {
        let mut file =
            File::open(&path).map_err(|e| vortex_err!("Failed to open file {:?}: {}", path, e))?;

        let file_size = file
            .seek(SeekFrom::End(0))
            .map_err(|e| vortex_err!("Failed to get file size: {}", e))?;

        Ok(Self {
            session,
            path,
            file,
            file_size,
        })
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

    async fn read_footer(&mut self) -> VortexResult<Footer> {
        Ok(self
            .session
            .open_options()
            .open_path(self.path.as_path())
            .await?
            .footer()
            .clone())
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
struct FooterSegments(Footer);

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

        if self.postscript_size > MAX_POSTSCRIPT_SIZE {
            println!(
                "WARNING: Postscript size exceeds maximum ({} > {})",
                self.postscript_size, MAX_POSTSCRIPT_SIZE
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
        println!("Total segments: {}", self.0.segment_map().len());
        let total_size = self
            .0
            .segment_map()
            .iter()
            .map(|s| s.length as u64)
            .sum::<u64>();
        println!("Total data size: {} bytes", total_size);

        println!("\nSegment details:\n");

        let segment_map = Arc::clone(self.0.segment_map());
        if segment_map.is_empty() {
            println!("<no segments>");
            return;
        }

        let mut segment_paths: Vec<Option<Vec<Arc<str>>>> = vec![None; segment_map.len()];
        let root_layout = Arc::clone(self.0.layout());

        let mut queue =
            VecDeque::<(Vec<Arc<str>>, LayoutRef)>::from_iter([(Vec::new(), root_layout)]);
        while !queue.is_empty() {
            let (path, layout) = queue.pop_front().vortex_expect("queue is not empty");
            for segment in layout.segment_ids() {
                segment_paths[*segment as usize] = Some(path.clone());
            }

            for (child_layout, child_name) in layout
                .children()
                .vortex_expect("Failed to deserialize children")
                .into_iter()
                .zip(layout.child_names())
            {
                let child_path = path.iter().cloned().chain([child_name]).collect();
                queue.push_back((child_path, child_layout));
            }
        }

        // Find the largest values for formatting
        let max_offset = segment_map.last().vortex_expect("non-empty").offset;
        let max_length = segment_map
            .iter()
            .map(|s| s.length)
            .max()
            .vortex_expect("non-empty");
        let max_alignment = segment_map
            .iter()
            .map(|s| s.alignment)
            .max()
            .vortex_expect("non-empty");

        // Calculate all widths
        let offset_width = max_offset.to_string().len();
        let end_width = (max_offset + max_length as u64).to_string().len();
        let length_width = max_length.to_string().len().max(6);
        let alignment_width = max_alignment.to_string().len().max(5);
        let index_width = segment_paths.len().to_string().len();

        // Print header
        println!(
            "{:>index_w$}  {:>offset_w$}..{:<end_w$}  {:>length_w$}  {:>align_w$}  Path",
            "#",
            "Start",
            "End",
            "Length",
            "Align",
            index_w = index_width,
            offset_w = offset_width,
            end_w = end_width,
            length_w = length_width,
            align_w = alignment_width,
        );

        for (i, name) in segment_paths.iter().enumerate() {
            let segment = &segment_map[i];
            let end_offset = segment.offset + segment.length as u64;

            print!(
                "{:>index_w$}  {:>offset_w$}..{:<end_w$}  ",
                i,
                segment.offset,
                end_offset,
                index_w = index_width,
                offset_w = offset_width,
                end_w = end_width,
            );
            print!(
                "{:>length_w$}  {:>align_w$}  ",
                segment.length,
                *segment.alignment,
                length_w = length_width,
                align_w = alignment_width,
            );
            println!(
                "{}",
                match name.as_ref() {
                    Some(path) => format!("{}", path.iter().format(".")),
                    None => "<missing>".to_string(),
                }
            );
        }
    }
}
