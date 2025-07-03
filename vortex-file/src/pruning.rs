// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::arrays::{ConstantArray, StructArray};
use vortex_array::stats::{Stat, StatsProvider, StatsSet};
use vortex_array::validity::Validity;
use vortex_dtype::{Field, FieldName, FieldNames, FieldPath, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_expr::pruning::field_path_stat_field_name;
use vortex_scalar::Scalar;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

pub fn extract_relevant_file_stats_as_struct_row(
    access: &HashMap<FieldPath, HashSet<Stat>>,
    stats_sets: &Arc<[StatsSet]>,
    struct_dtype: &StructFields,
) -> VortexResult<Option<ArrayRef>> {
    if access.is_empty() {
        return StructArray::try_new(FieldNames::default(), vec![], 1, Validity::NonNullable)
            .map(|s| Some(s.to_array()));
    }

    let mut columns: Vec<(FieldName, ArrayRef)> = Vec::with_capacity(access.len() * 2);
    for (field_path, stats) in access.into_iter() {
        if field_path.path().len() != 1 {
            return Ok(None);
        }
        let Field::Name(field) = &field_path.path()[0] else {
            return Ok(None);
        };

        let field_idx = struct_dtype
            .find(field)
            .ok_or_else(|| vortex_err!("Missing field: {field}"))?;
        let field_dtype = struct_dtype
            .field_by_index(field_idx)
            .vortex_expect("Field must exist");

        let Some(stat_set) = stats_sets.get(field_idx) else {
            vortex_bail!("missing stat field {} from stats set", field)
        };

        for stat in stats {
            let Some(stat_value) = stat_set.get(*stat).and_then(|p| p.as_exact()) else {
                vortex_bail!("missing stat {}, {} from stats set", field, stat)
            };
            if stat == &Stat::Max || stat == &Stat::Min {
                columns.push((
                    field_path_stat_field_name(field_path, *stat),
                    ConstantArray::new(Scalar::new(field_dtype.clone(), stat_value.clone()), 1)
                        .to_array(),
                ))
            } else {
                todo!("unsupported file prune stat {stat}")
            }
        }
    }
    Ok(Some(
        StructArray::from_fields(columns.as_slice())?.to_array(),
    ))
}
