use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use vortex::error::{VortexExpect, VortexResult};
use vortex::file::VortexOpenOptions;

pub async fn segments(file: impl AsRef<Path>) -> VortexResult<()> {
    let vxf = VortexOpenOptions::file().open(file).await?;

    let segment_map = vxf.footer().segment_map();
    let segment_source = vxf.segment_source();

    let mut segment_names: Vec<Option<Arc<str>>> = vec![None; segment_map.len()];

    let root_reader =
        vxf.footer()
            .layout()
            .new_reader(&"".into(), &segment_source, vxf.footer().ctx())?;

    let mut queue = VecDeque::from_iter([root_reader]);
    while !queue.is_empty() {
        let reader = queue.pop_front().vortex_expect("queue is not empty");
        for segment in reader.segment_ids() {
            segment_names[*segment as usize] = Some(reader.name().clone());
        }

        for (child_layout, child_name) in reader.children()?.iter().zip(reader.child_names()) {
            queue.push_back(child_layout.new_reader(
                &child_name,
                &segment_source,
                vxf.footer().ctx(),
            )?);
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
