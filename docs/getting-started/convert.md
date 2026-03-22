# Convert

If you haven't already, download the sample data (see [Install](install.md#sample-data)):

```bash
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

The `vx convert` command converts a Parquet file to Vortex, applying compression automatically:

```bash
vx convert yellow_tripdata_2024-01.parquet
```

This produces `yellow_tripdata_2024-01.vortex` in the same directory. By default it uses
BtrBlocks compression, chunking on Parquet row-group boundaries.

## Compression strategies

Choose a compression strategy with `--strategy`:

```bash
# Default: BtrBlocks compressor
vx convert yellow_tripdata_2024-01.parquet --strategy btrblocks

# Compact: more aggressive compression
vx convert yellow_tripdata_2024-01.parquet --strategy compact
```
