use async_trait::async_trait;
use vortex::ArrayRef;

pub mod data_downloads;
pub mod public_bi_data;
pub mod struct_list_of_ints;
pub mod taxi_data;
pub mod tpch_l_comment;

#[async_trait]
pub trait BenchmarkDataset {
    fn name(&self) -> &str;

    async fn to_vortex_array(&self) -> ArrayRef;
}
