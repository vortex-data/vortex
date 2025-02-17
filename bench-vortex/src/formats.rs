use crate::Format;

pub fn parse_formats(formats: Vec<String>) -> Vec<Format> {
    formats
        .into_iter()
        .map(|format| match format.as_ref() {
            "arrow" => Format::Arrow,
            "parquet" => Format::Parquet,
            "vortex" => Format::OnDiskVortex,
            _ => panic!("unrecognized format: {}", format),
        })
        .collect::<Vec<_>>()
}
