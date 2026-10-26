// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DataSource;
use crate::DataSourceRef;
use crate::DataSourceScanRef;
use crate::ScanRequest;

struct LegacySource {
    dtype: DType,
}

#[async_trait]
impl DataSource for LegacySource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    async fn scan(&self, _request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        vortex_bail!("metadata lookup must not scan the source")
    }

    async fn field_statistics(&self, _path: &FieldPath) -> VortexResult<StatsSet> {
        let mut stats = StatsSet::of(Stat::Min, Precision::inexact(1_i32));
        stats.set(Stat::NullCount, Precision::exact(2_u64));
        Ok(stats)
    }
}

fn lookup(
    source: &dyn DataSource,
    path: &FieldPath,
    aggregate: &AggregateFnRef,
) -> VortexResult<Precision<Scalar>> {
    source
        .field_statistic(path, aggregate)
        .now_or_never()
        .expect("stored metadata is immediately available")
}

#[test]
fn legacy_statistics_preserve_precision_and_match_aggregate_options() -> VortexResult<()> {
    let source: DataSourceRef = Arc::new(LegacySource {
        dtype: DType::struct_(
            [("value", DType::Primitive(PType::I32, Nullability::Nullable))],
            Nullability::NonNullable,
        ),
    });
    let path = FieldPath::from_name("value");
    let min = Min.bind(NumericalAggregateOpts::skip_nans());
    let dtype = path.resolve(source.dtype().clone()).expect("field exists");
    let result_dtype = min.return_dtype(&dtype).expect("minimum supports i32");
    assert_eq!(
        lookup(source.as_ref(), &path, &min)?,
        Precision::Inexact(Scalar::from(1_i32).cast(&result_dtype)?)
    );
    assert_eq!(
        lookup(source.as_ref(), &path, &NullCount.bind(EmptyOptions))?,
        Precision::Exact(Scalar::from(2_u64))
    );
    assert_eq!(
        lookup(
            source.as_ref(),
            &path,
            &Min.bind(NumericalAggregateOpts::include_nans())
        )?,
        Precision::Absent
    );
    assert_eq!(
        lookup(
            source.as_ref(),
            &path,
            &MinMax.bind(NumericalAggregateOpts::skip_nans())
        )?,
        Precision::Absent
    );
    assert_eq!(
        lookup(source.as_ref(), &FieldPath::from_name("missing"), &min)?,
        Precision::Absent
    );
    Ok(())
}
