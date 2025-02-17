use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use bzip2::read::BzDecoder;
use log::info;
use vortex::error::VortexError;

use crate::idempotent;

pub fn download_data(fname: PathBuf, data_url: &str) -> PathBuf {
    idempotent(&fname, |path| {
        info!("Downloading {} from {}", fname.to_str().unwrap(), data_url);
        let mut file = File::create(path).unwrap();
        let mut response = reqwest::blocking::get(data_url).unwrap();
        if !response.status().is_success() {
            panic!("Failed to download data from {}", data_url);
        }
        response.copy_to(&mut file)
    })
    .unwrap()
}

pub fn decompress_bz2(input_path: PathBuf, output_path: PathBuf) -> PathBuf {
    idempotent(&output_path, |path| {
        info!(
            "Decompressing bzip from {} to {}",
            input_path.to_str().unwrap(),
            output_path.to_str().unwrap()
        );
        let input_file = File::open(input_path).unwrap();
        let mut decoder = BzDecoder::new(input_file);

        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).unwrap();

        let mut output_file = File::create(path).unwrap();
        output_file.write_all(&buffer).unwrap();
        Ok::<PathBuf, VortexError>(output_path.clone())
    })
    .unwrap()
}
