// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;

use itertools::Itertools as _;
use vortex::error::{VortexExpect, VortexResult};
use vortex::file::VortexOpenOptions;
use vortex::layout::LayoutRef;

pub async fn segments(file: impl AsRef<Path>) -> VortexResult<()> {
    let vxf = VortexOpenOptions::file().open(file).await?;

    let segment_map = vxf.footer().segment_map();

    let mut segment_paths: Vec<Option<Vec<Arc<str>>>> = vec![None; segment_map.len()];

    let root_layout = vxf.footer().layout().clone();

    let mut queue = VecDeque::<(Vec<Arc<str>>, LayoutRef)>::from_iter([(Vec::new(), root_layout)]);
    while !queue.is_empty() {
        let (path, layout) = queue.pop_front().vortex_expect("queue is not empty");
        for segment in layout.segment_ids() {
            segment_paths[*segment as usize] = Some(path.clone());
        }

        for (child_layout, child_name) in layout.children()?.into_iter().zip(layout.child_names()) {
            let child_path = path.iter().cloned().chain([child_name]).collect();
            queue.push_back((child_path, child_layout));
        }
    }

    for (i, name) in segment_paths.iter().enumerate() {
        println!(
            "{}: {}..{} (len={}, alignment={}) - {}",
            i,
            segment_map[i].offset,
            segment_map[i].offset + segment_map[i].length as u64,
            segment_map[i].length,
            segment_map[i].alignment,
            match name.as_ref() {
                Some(path) => &path.iter().format(".") as &dyn Display,
                None => &"<missing>",
            }
        );
    }

    Ok(())
}
