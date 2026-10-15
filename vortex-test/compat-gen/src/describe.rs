// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Print what each `.vortex` file in a directory contains: its dtype, the layout IDs in its
//! layout tree and the array encoding IDs in its data. For triaging old-reader failures.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use vortex::VortexSessionDefault;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::adapter;

fn describe_one(path: &Path) -> VortexResult<String> {
    let bytes = std::fs::read(path).map_err(|e| vortex_err!("read {}: {e}", path.display()))?;
    let bytes = ByteBuffer::from(bytes);
    let session = VortexSession::default().with_tokio();
    let file = session.open_options().open_buffer(bytes.clone())?;
    let dtype = file.dtype().clone();
    let mut layouts = BTreeSet::new();
    for layout in Arc::clone(file.footer().layout()).depth_first_traversal() {
        layouts.insert(layout?.dyn_encoding_id().to_string());
    }
    let array = adapter::read_file(bytes)?;
    let mut encodings = BTreeSet::new();
    for node in array.depth_first_traversal() {
        encodings.insert(node.encoding_id().to_string());
    }
    Ok(format!(
        "rows={} dtype={dtype}\n    layouts: {}\n    arrays:  {}",
        array.len(),
        layouts.into_iter().collect::<Vec<_>>().join(", "),
        encodings.into_iter().collect::<Vec<_>>().join(", ")
    ))
}

/// Describe every `.vortex` file in `dir`, or only those whose name contains `filter`.
pub fn describe(dir: &Path, filter: Option<&str>) -> VortexResult<()> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| vortex_err!("failed to read dir {}: {e}", dir.display()))?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().to_string_lossy().to_string();
            (name.ends_with(".vortex") && filter.is_none_or(|f| name.contains(f))).then_some(name)
        })
        .collect();
    names.sort();
    for name in names {
        match describe_one(&dir.join(&name)) {
            Ok(text) => println!("{name}: {text}"),
            Err(e) => println!("{name}: ERROR {e}"),
        }
    }
    Ok(())
}
