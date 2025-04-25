use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use vortex::error::{VortexExpect, VortexResult};
use vortex::file::VortexOpenOptions;

pub async fn segments(file: impl AsRef<Path>) -> VortexResult<()> {
    let vxf = VortexOpenOptions::file().open(file).await?;

    let segment_map = vxf.footer().segment_map();

    let mut segment_names: Vec<Option<Arc<str>>> = vec![None; segment_map.len()];

    let mut queue = VecDeque::from_iter([vxf.layout_reader()?]);
    while !queue.is_empty() {
        let reader = queue.pop_front().vortex_expect("queue is not empty");
        for segment in reader.layout().segments() {
            segment_names[*segment as usize] = Some(reader.layout().name().clone());
        }
        queue.extend(reader.children()?);
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
