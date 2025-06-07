//! File reading example.

use std::env::args;
use std::error::Error;

use vortex::session::VortexSessionBuilder;
use vortex_array::ArrayRegistryBuilder;
use vortex_file::ArrayRegistryExt;
use vortex_layout::{LayoutRegistryBuilder, LayoutRegistryExt};

#[allow(clippy::use_debug, clippy::expect_used)]
#[tokio::main]
pub async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let session = VortexSessionBuilder::new()
        .with_encodings(ArrayRegistryBuilder::full())
        .with_layouts(LayoutRegistryBuilder::full())
        .build();

    let path = args().nth(1).expect("path to vortex file must be provided");

    let vx = session.open(path).await?;
    for batch in vx.scan()?.into_array_iter()? {
        println!("next batch: {:?}", batch?);
    }

    Ok(())
}
