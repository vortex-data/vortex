// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::path::PathBuf;

use tracing_subscriber::EnvFilter;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_scan_v2::ScanBuilder;

fn main() -> VortexResult<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("vortex_scan_v2=debug"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .without_time()
        .init();

    let path = env::args_os().nth(1).map_or_else(
        || PathBuf::from("vortex-bench/data/tpch/0.01/vortex-file-compressed/lineitem.vortex"),
        PathBuf::from,
    );

    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);
        vortex_file::register_default_encodings(&session);

        let file = session.open_options().open_path(&path).await?;
        println!(
            "opened {}: rows={}, dtype={}",
            path.display(),
            file.row_count(),
            file.dtype()
        );

        let filter = gt(get_item("l_linenumber", root()), lit(5_i32));
        let projection = select(["l_orderkey", "l_linenumber"], root());
        let result = ScanBuilder::try_new(
            file.footer().layout(),
            file.segment_source(),
            session.clone(),
        )?
        .with_filter(filter.clone())
        .with_projection(projection.clone())
        .into_array_stream()?
        .read_all()
        .await?;

        println!(
            "scan result: rows={}, dtype={}",
            result.len(),
            result.dtype()
        );
        let expected = file
            .scan()?
            .with_filter(filter.bind(file.dtype())?)
            .with_projection(projection.bind(file.dtype())?)
            .into_array_stream()?
            .read_all()
            .await?;
        assert_arrays_eq!(result, expected, &mut session.create_execution_ctx());
        println!("validated every result value against the LayoutReader scan");
        Ok(())
    })
}
