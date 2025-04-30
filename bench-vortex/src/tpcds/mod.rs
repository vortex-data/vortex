use datafusion::prelude::SessionContext;
use url::Url;

use crate::Format;
use crate::df::{get_session_context, make_object_store};

pub async fn load_datasets(_base_dir: &Url, _format: Format) -> anyhow::Result<SessionContext> {
    let context = get_session_context(true);
    Ok(context)
}

pub fn tpcds_queries() -> impl Iterator<Item = (usize, Vec<String>)> {
    todo!()
}
