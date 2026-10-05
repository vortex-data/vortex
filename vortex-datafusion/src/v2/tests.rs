// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_datasource::source::DataSource as _;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::aggregate_fn::AggregateFnVTableExt;
use vortex::array::aggregate_fn::NumericalAggregateOpts;
use vortex::array::aggregate_fn::fns::min::Min;
use vortex::array::stats::AggregateResults;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::StructFields;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::stats::Precision;
use vortex::scalar::Scalar;
use vortex::scan::DataSource;
use vortex::scan::DataSourceScanRef;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;

use super::VortexDataSource;

struct StatisticsSource {
    dtype: DType,
    min: Precision<i32>,
    fail: bool,
}

impl StatisticsSource {
    fn new(min: Precision<i32>, fail: bool) -> Self {
        Self {
            dtype: DType::Struct(
                StructFields::new(["a", "b"].into(), vec![PType::I32.into(); 2]),
                Nullability::NonNullable,
            ),
            min,
            fail,
        }
    }
}

#[async_trait]
impl DataSource for StatisticsSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    async fn scan(&self, _request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        vortex_bail!("Statistics planning must not scan the source")
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<AggregateResults> {
        if self.fail {
            vortex_bail!("Field statistics unavailable")
        }

        let min = if field_path == &FieldPath::from_name("a") {
            self.min
        } else if field_path == &FieldPath::from_name("b") {
            Precision::Exact(42)
        } else {
            vortex_bail!("Unknown field: {field_path}")
        };

        AggregateResults::try_new(
            &DType::from(PType::I32),
            [(
                Min.bind(NumericalAggregateOpts::skip_nans()),
                min.map(|value| Scalar::primitive(value, Nullability::Nullable)),
            )],
        )
    }
}

#[rstest]
#[case::exact(Precision::Exact(7), DFPrecision::Exact(ScalarValue::Int32(Some(7))))]
#[case::inexact(
    Precision::Inexact(7),
    DFPrecision::Inexact(ScalarValue::Int32(Some(7)))
)]
#[case::missing(Precision::Absent, DFPrecision::Absent)]
#[tokio::test]
async fn source_statistics_keep_precision(
    #[case] min: Precision<i32>,
    #[case] expected: DFPrecision<ScalarValue>,
) -> anyhow::Result<()> {
    let source = Arc::new(StatisticsSource::new(min, false));
    let adapter = VortexDataSource::builder(source, VortexSession::default())
        .build()
        .await?;
    let statistics = adapter.partition_statistics(None)?;

    assert_eq!(statistics.column_statistics[0].min_value, expected);
    assert_eq!(
        statistics.column_statistics[0].max_value,
        DFPrecision::Absent
    );
    assert_eq!(statistics.num_rows, DFPrecision::Absent);
    Ok(())
}

#[tokio::test]
async fn projected_statistics_follow_field_names() -> anyhow::Result<()> {
    let source = Arc::new(StatisticsSource::new(Precision::Exact(7), false));
    let adapter = VortexDataSource::builder(source, VortexSession::default())
        .with_projection(vec![1, 0])
        .build()
        .await?;
    let statistics = adapter.partition_statistics(None)?;

    assert_eq!(statistics.column_statistics.len(), 2);
    assert_eq!(
        statistics.column_statistics[0].min_value,
        DFPrecision::Exact(ScalarValue::Int32(Some(42)))
    );
    assert_eq!(
        statistics.column_statistics[1].min_value,
        DFPrecision::Exact(ScalarValue::Int32(Some(7)))
    );
    Ok(())
}

#[tokio::test]
async fn source_statistics_errors_reach_the_builder() {
    let source = Arc::new(StatisticsSource::new(Precision::Absent, true));
    let result = VortexDataSource::builder(source, VortexSession::default())
        .build()
        .await;

    assert!(result.is_err_and(|error| error.to_string().contains("Field statistics unavailable")));
}
