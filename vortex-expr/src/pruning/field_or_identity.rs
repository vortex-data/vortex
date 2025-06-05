use std::fmt::Display;

use vortex_array::stats::Stat;
use vortex_dtype::FieldName;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FieldOrIdentity {
    Field(FieldName),
    Identity,
}

pub fn stat_field_name(field: &FieldName, stat: Stat) -> FieldName {
    FieldName::from(stat_field_name_string(field, stat))
}

pub(crate) fn stat_field_name_string(field: &FieldName, stat: Stat) -> String {
    format!("{field}_{stat}")
}

impl FieldOrIdentity {
    pub(crate) fn stat_field_name(&self, stat: Stat) -> FieldName {
        FieldName::from(self.stat_field_name_string(stat))
    }

    pub(crate) fn stat_field_name_string(&self, stat: Stat) -> String {
        match self {
            FieldOrIdentity::Field(field) => stat_field_name_string(field, stat),
            FieldOrIdentity::Identity => stat.to_string(),
        }
    }
}

impl Display for FieldOrIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldOrIdentity::Field(field) => write!(f, "{field}"),
            FieldOrIdentity::Identity => write!(f, "$[]"),
        }
    }
}

impl<T> From<T> for FieldOrIdentity
where
    FieldName: From<T>,
{
    fn from(value: T) -> Self {
        FieldOrIdentity::Field(FieldName::from(value))
    }
}
