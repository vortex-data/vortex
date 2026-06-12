// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::pruning::field_path_stat_field_name;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
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
            .map(|s| Some(s.into_array()));
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
            let Some(stat_value) = typed_stats.get(*stat).as_exact() else {
                vortex_bail!("missing stat {}, {} from stats set", field, stat)
            };
            columns.push((
                field_path_stat_field_name(field_path, *stat),
                ConstantArray::new(stat_value, 1).into_array(),
            ));
        }
    }
    // Every accessible field may still carry an empty stats set (e.g. dtypes that support no
    // file stats), in which case the scope is the empty struct and the row must match it.
    if columns.is_empty() {
        return StructArray::try_new(FieldNames::default(), vec![], 1, Validity::NonNullable)
            .map(|s| Some(s.into_array()));
    }
    columns.sort_by(|(left, _), (right, _)| left.cmp(right));

    Ok(Some(
        StructArray::from_fields(columns.as_slice())?.into_array(),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::stats::StatsSet;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::extract_relevant_file_stats_as_struct_row;

    /// Fields whose stats sets are all empty must yield the 1-row empty struct (matching the
    /// empty bound stats scope) rather than erroring on `StructArray::from_fields(&[])`.
    #[test]
    fn empty_stat_sets_yield_empty_row() -> VortexResult<()> {
        let access = HashMap::from_iter([(FieldPath::from_name("a"), HashSet::default())]);
        let stats_sets: Arc<[StatsSet]> = vec![StatsSet::default()].into();
        let fields = StructFields::from_iter([(
            "a",
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )]);

        let row = extract_relevant_file_stats_as_struct_row(&access, &stats_sets, &fields)?
            .ok_or_else(|| vortex_err!("expected a stats row"))?;
        assert_eq!(row.len(), 1);
        assert!(
            row.dtype()
                .as_struct_fields_opt()
                .is_some_and(|f| f.nfields() == 0)
        );
        Ok(())
    }
}
