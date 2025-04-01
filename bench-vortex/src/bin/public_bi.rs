use bench_vortex::public_bi::{PBI_DATASETS, PBIDataset};
use tokio::runtime::Builder;

fn main() -> anyhow::Result<()> {
    let hatred = PBI_DATASETS.get(PBIDataset::TrainsUK1);
    println!("{:?}", hatred.queries());
    let dataset = hatred.dataset().unwrap();
    let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
    runtime.block_on(dataset.write_as_vortex());
    Ok(())
}
