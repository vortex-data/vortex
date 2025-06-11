use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::arrays::{ConstantArray, StructArray};
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::validity::Validity;
use vortex_dtype::{Field, StructFields};
use vortex_error::VortexResult;
use vortex_expr::pruning::{PruningPredicate, access_path_stat_field_name};
use vortex_scalar::Scalar;

pub fn extract_relevant_stat_as_struct_row(
    predicate: &PruningPredicate,
    stats_set: &Arc<[StatsSet]>,
    struct_dtype: &Arc<StructFields>,
) -> VortexResult<Option<ArrayRef>> {
    if predicate.required_stats().is_empty() {
        return StructArray::try_new([].into(), vec![], 1, Validity::NonNullable)
            .map(|s| Some(s.to_array()));
    }
    let mut columns = vec![];
    for (access_path, stats) in predicate.required_stats() {
        if access_path.field_path().path().len() != 1 {
            return Ok(None);
        }
        let Field::Name(field) = &access_path.field_path().path()[0] else {
            return Ok(None);
        };

        let field_idx = struct_dtype.find(field)?;
        let field_dtype = struct_dtype.field_by_index(field_idx)?;
        let Some(cols) = stats_set[field_idx]
            .iter()
            .filter(|(stat, _)| stats.contains(stat))
            .map(|(stat, value)| {
                value.as_ref().as_exact().and_then(|value| {
                    (stat == &Stat::Max || stat == &Stat::Min).then(|| {
                        (
                            access_path_stat_field_name(access_path, *stat),
                            ConstantArray::new(Scalar::new(field_dtype.clone(), value.clone()), 1)
                                .to_array(),
                        )
                    })
                })
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };

        columns.extend(cols)
    }

    StructArray::from_fields(&columns).map(|s| Some(s.to_array()))
}
