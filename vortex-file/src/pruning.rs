// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::expr::pruning::field_path_stat_field_name;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_dtype::Field;
use vortex_dtype::FieldName;
use vortex_dtype::FieldNames;
use vortex_dtype::FieldPath;
use vortex_dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
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
        if field_path.parts().len() != 1 {
            return Ok(None);
        }
        let Field::Name(field) = &field_path.parts()[0] else {
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
        let typed_stats = stat_set.as_typed_ref(&field_dtype);

        for stat in stats {
            if matches!(
                stat,
                Stat::Max | Stat::Min | Stat::NaNCount | Stat::NullCount
            ) {
                let Some(stat_value) = typed_stats.get(*stat).and_then(|p| p.as_exact()) else {
                    vortex_bail!("missing stat {}, {} from stats set", field, stat)
                };
                columns.push((
                    field_path_stat_field_name(field_path, *stat),
                    ConstantArray::new(stat_value, 1).to_array(),
                ));
            } else {
                todo!("unsupported file prune stat {stat}")
            }
        }
    }
    Ok(Some(
        StructArray::from_fields(columns.as_slice())?.to_array(),
    ))
}
