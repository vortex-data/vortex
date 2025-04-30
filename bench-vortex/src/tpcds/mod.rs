use std::fs;
use std::path::Path;

use datafusion::prelude::SessionContext;
use url::Url;

use crate::Format;
use crate::df::get_session_context;

pub async fn load_datasets(_base_dir: &Url, _format: Format) -> anyhow::Result<SessionContext> {
    let context = get_session_context(true);
    Ok(context)
}

pub fn tpcds_queries() -> impl Iterator<Item = (usize, String)> {
    (1..=99).map(|idx| (idx, tpch_query(idx)))
}

// A few tpch queries have multiple statements, this handles that
fn tpch_query(query_idx: usize) -> String {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tpcds")
        .join(format!("{:02}", query_idx))
        .with_extension("sql");
    println!("dir {:?}", manifest_dir.to_str());
    fs::read_to_string(manifest_dir).unwrap()
}
