use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use vortex::error::{VortexExpect, VortexResult};
use vortex::file::VortexOpenOptions;
use vortex_layout::LayoutRef;

pub async fn segments(file: impl AsRef<Path>) -> VortexResult<()> {
    let vxf = VortexOpenOptions::file().open(file).await?;

    let segment_map = vxf.footer().segment_map();

    let mut segment_names: Vec<Option<Arc<str>>> = vec![None; segment_map.len()];

    let root_layout = vxf.footer().layout().clone();

    let mut queue = VecDeque::<(Arc<str>, LayoutRef)>::from_iter([("".into(), root_layout)]);
    while !queue.is_empty() {
        let (name, layout) = queue.pop_front().vortex_expect("queue is not empty");
        for segment in layout.segment_ids() {
            segment_names[*segment as usize] = Some(name.clone());
        }

        for (child_layout, child_name) in layout.children()?.into_iter().zip(layout.child_names()) {
            queue.push_back((child_name, child_layout));
        }
    }

    for (i, name) in segment_names.iter().enumerate() {
        println!(
            "{}: {}..{} (len={}, alignment={}) - {}",
            i,
            segment_map[i].offset,
            segment_map[i].offset + segment_map[i].length as u64,
            segment_map[i].length,
            segment_map[i].alignment,
            name.clone().unwrap_or_else(|| "<missing>".into())
        );
    }

    Ok(())
}
