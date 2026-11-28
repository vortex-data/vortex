// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measure a per-file template dictionary against the original layout FlatBuffer.
//! Pass one or more .vortex files. Each output row describes one round-tripped file.

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;

use prost::Message;
use vortex_file::EOF_SIZE;
use vortex_file::MAGIC_BYTES;
use vortex_file::VERSION;
use vortex_file::flatbuffers::footer::Footer;
use vortex_file::flatbuffers::footer::Postscript;
use vortex_file::flatbuffers::footer::PostscriptSegment;
use vortex_layout::flatbuffers::layout::Layout;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, PartialEq, Eq, Hash, Message)]
struct Template {
    #[prost(uint32, tag = "1")]
    encoding: u32,
    #[prost(bytes = "vec", optional, tag = "2")]
    metadata: Option<Vec<u8>>,
    #[prost(uint32, tag = "3")]
    segment_slots: u32,
    #[prost(uint32, repeated, tag = "4")]
    children: Vec<u32>,
    #[prost(uint32, optional, tag = "5")]
    repeat_child: Option<u32>,
}

#[derive(Clone, PartialEq, Message)]
struct PackedLayout {
    #[prost(message, repeated, tag = "1")]
    templates: Vec<Template>,
    #[prost(uint32, tag = "2")]
    root: u32,
    // Bindings are in depth-first, left-to-right order.
    #[prost(uint64, repeated, tag = "3")]
    rows: Vec<u64>,
    #[prost(uint32, repeated, tag = "4")]
    segments: Vec<u32>,
    #[prost(uint32, repeated, tag = "5")]
    repeat_counts: Vec<u32>,
    // When present, rows contains one value per run.
    #[prost(uint32, repeated, tag = "6")]
    row_runs: Vec<u32>,
}

#[derive(Default)]
struct Encoder {
    ids: HashMap<Template, u32>,
    packed: PackedLayout,
}

impl Encoder {
    fn intern(&mut self, node: Layout<'_>, chunked: Option<u16>) -> u32 {
        let mut children = Vec::new();
        let mut repeat_child = None;
        for (index, child) in node.children().into_iter().flatten().enumerate() {
            let id = self.intern(child, chunked);
            if index == 0 && Some(node.encoding()) == chunked {
                repeat_child = Some(id);
            } else if repeat_child != Some(id) {
                if let Some(previous) = repeat_child.take() {
                    children.resize(index, previous);
                }
                children.push(id);
            }
        }
        let key = Template {
            encoding: node.encoding().into(),
            metadata: node.metadata().map(|bytes| bytes.bytes().to_vec()),
            segment_slots: node.segments().map_or(0, |ids| ids.len() as u32),
            children,
            repeat_child,
        };
        *self.ids.entry(key).or_insert_with_key(|key| {
            let id = self.packed.templates.len() as u32;
            self.packed.templates.push(key.clone());
            id
        })
    }
}

fn bind(packed: &mut PackedLayout, id: u32, node: Layout<'_>) {
    packed.rows.push(node.row_count());
    packed.segments.extend(node.segments().into_iter().flatten());
    let children = node.children().into_iter().flatten();
    let repeat = packed.templates[id as usize].repeat_child;
    if repeat.is_some() {
        packed.repeat_counts.push(node.children().map_or(0, |v| v.len() as u32));
    }
    for (index, child) in children.enumerate() {
        let child_id = repeat.unwrap_or_else(|| packed.templates[id as usize].children[index]);
        bind(packed, child_id, child);
    }
}

// Compare the decoded wire message with the input. Consume every slot, including
// repeated child counts, to detect ordering mistakes even when file sizes agree.
fn verify(packed: &PackedLayout, root: Layout<'_>) -> Result<()> {
    fn visit(
        templates: &[Template],
        id: u32,
        node: Layout<'_>,
        rows: &mut impl Iterator<Item = u64>,
        segments: &mut impl Iterator<Item = u32>,
        counts: &mut impl Iterator<Item = u32>,
    ) -> Result<()> {
        let t = templates.get(id as usize).ok_or("missing template")?;
        let children = node.children();
        let nchildren = children.map_or(0, |v| v.len());
        if t.encoding != u32::from(node.encoding())
            || t.metadata.as_deref() != node.metadata().map(|v| v.bytes())
            || rows.next() != Some(node.row_count())
            || t.segment_slots as usize != node.segments().map_or(0, |v| v.len())
            || node.segments().into_iter().flatten().any(|id| segments.next() != Some(id))
        {
            return Err("layout binding mismatch".into());
        }
        let arity = match t.repeat_child {
            Some(_) => counts.next().ok_or("missing repeat count")? as usize,
            None => t.children.len(),
        };
        if arity != nchildren {
            return Err("child count mismatch".into());
        }
        for (i, child) in children.into_iter().flatten().enumerate() {
            visit(templates, t.repeat_child.unwrap_or_else(|| t.children[i]), child, rows, segments, counts)?;
        }
        Ok(())
    }
    let mut rows = packed.rows.iter().enumerate().flat_map(|(i, &value)| {
        std::iter::repeat_n(value, packed.row_runs.get(i).copied().unwrap_or(1) as usize)
    });
    let mut segments = packed.segments.iter().copied();
    let mut counts = packed.repeat_counts.iter().copied();
    visit(&packed.templates, packed.root, root, &mut rows, &mut segments, &mut counts)?;
    if rows.next().is_some() || segments.next().is_some() || counts.next().is_some() {
        return Err("unused bindings".into());
    }
    Ok(())
}

fn section(file: &mut File, locator: PostscriptSegment<'_>, end: u64) -> Result<Vec<u8>> {
    if locator._compression().is_some() || locator._encryption().is_some() {
        return Err("unsupported transformed footer segment".into());
    }
    if locator.offset().checked_add(locator.length().into()).is_none_or(|offset| offset > end) {
        return Err("footer segment out of bounds".into());
    }
    file.seek(SeekFrom::Start(locator.offset()))?;
    let mut bytes = vec![0; locator.length() as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn measure(path: &Path, locate_only: bool) -> Result<()> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    file.seek(SeekFrom::End(-(EOF_SIZE as i64)))?;
    let mut trailer = [0; EOF_SIZE];
    file.read_exact(&mut trailer)?;
    if trailer[4..] != MAGIC_BYTES || trailer[..2] != VERSION.to_le_bytes() {
        return Err("unsupported Vortex trailer".into());
    }
    let ps_len = u16::from_le_bytes([trailer[2], trailer[3]]) as usize;
    let ps_start = size.checked_sub((EOF_SIZE + ps_len) as u64).ok_or("invalid postscript length")?;
    file.seek(SeekFrom::Start(ps_start))?;
    let mut bytes = vec![0; ps_len];
    file.read_exact(&mut bytes)?;
    let ps = flatbuffers::root::<Postscript>(&bytes)?;
    let footer = ps.footer().ok_or("missing footer")?;
    let layout = ps.layout().ok_or("missing layout")?;
    let start = [Some(footer), Some(layout), ps.dtype(), ps.statistics()]
        .into_iter().flatten()
        .chain(ps.metadata().into_iter().flatten().map(|entry| entry.segment()))
        .map(|s| s.offset()).min().ok_or("empty footer")?;
    if locate_only {
        println!("{start}");
        return Ok(());
    }
    let registry = section(&mut file, footer, ps_start)?;
    let specs = flatbuffers::root::<Footer>(&registry)?.layout_specs().ok_or("missing layout specs")?;
    let chunked = specs.iter().position(|spec| spec.id() == "vortex.chunked").map(u16::try_from).transpose()?;
    let layout_bytes = section(&mut file, layout, ps_start)?;
    let root = flatbuffers::root::<Layout>(&layout_bytes)?;
    let mut encoder = Encoder::default();
    let id = encoder.intern(root, chunked);
    let mut packed = encoder.packed;
    packed.root = id;
    bind(&mut packed, id, root);
    let nodes = packed.rows.len();
    let plain = packed.encode_to_vec();
    let original_rows = std::mem::take(&mut packed.rows);
    for run in original_rows.chunk_by(|a, b| a == b) {
        packed.rows.push(run[0]);
        packed.row_runs.push(u32::try_from(run.len())?);
    }
    let encoded = if packed.encoded_len() < plain.len() { packed.encode_to_vec() } else { plain };
    verify(&PackedLayout::decode(encoded.as_slice())?, root)?;
    println!("{}\t{size}\t{}\t{}\t{}\t{nodes}\t{}", path.display(), size - start, layout_bytes.len(), encoded.len(), packed.templates.len());
    Ok(())
}

fn main() -> Result<()> {
    let mut paths = env::args_os().skip(1).peekable();
    let locate_only = paths.next_if(|arg| arg == "--footer-start").is_some();
    if paths.peek().is_none() {
        return Err("usage: layout_templates FILE.vortex [...]".into());
    }
    for path in paths {
        let path = Path::new(&path);
        measure(path, locate_only).map_err(|error| format!("{}: {error}", path.display()))?;
    }
    Ok(())
}
