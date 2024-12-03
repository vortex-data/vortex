# Install a JRE 11: https://adoptium.net/temurin/releases/
#
# install the gcs conenctor:
#
#     curl https://raw.githubusercontent.com/broadinstitute/install-gcs-connector/master/install_gcs_connector.py | uv run python3
#
# Run like this:
#
#     PYSPARK_SUBMIT_ARGS="--driver-memory 8g --executor-memory 8g pyspark-shell" uv run python3 import.py

import os

import vortex
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

if not os.path.exists('tiny-no-lists-of-lists.vcf.parquet'):
    print('writing: tiny-no-lists-of-lists.vcf.parquet')
    import hail as hl
    hl.init(master='local[*]')
    hl.default_reference('GRCh38')

    mt = hl.read_matrix_table('gs://gcp-public-data--gnomad/release/3.1.2/mt/genomes/gnomad.genomes.v3.1.2.hgdp_1kg_subset_dense.mt')
    mt = mt.head(n_rows=5)
    mt = mt.select_rows() # remove all row metadata (future work!)
    mt = mt.key_rows_by() # demote row key columns to normal columns
    mt = mt.select_rows(
        # demote locus structure to simple columns for ease of arrow conversion
        chromosome=mt.locus.contig,
        position=mt.locus.position,
    )
    mt = mt.localize_entries('entries') # convert matrix into table with a list of entries per row
    mt = mt.select(
        'chromosome',
        'position',
        # convert list of struct to struct of list
        GT=mt.entries.map(lambda entry: entry.GT.n_alt_alleles()),  # convert from extension data type to a number
        GQ=mt.entries.GQ,
        # # list of lists fields:
        # PL=mt.entries.PL,
        # AD=mt.entries.AD,
    )

    df = mt.to_pandas()

    df = pa.Table.from_pandas(df)
    pq.write_table(df, 'tiny-no-lists-of-lists.vcf.parquet')
else:
    print('found: tiny-no-lists-of-lists.vcf.parquet')

if not os.path.exists('tiny-no-lists-of-lists.vcf.vortex'):
    print('writing: tiny-no-lists-of-lists.vcf.parquet')
    vortex.io.write_path(
        vortex.encoding.compress(
            vortex.array(
                pa.Table.from_pandas(pd.read_parquet('tiny-no-lists-of-lists.vcf.parquet'))
            )
        ),
        'tiny-no-lists-of-lists.vcf.vortex'
    )
else:
    file = vortex.io.read_url('file:////Users/joeisaacs/git/spiraldb/vortex/vortex-genetics/tiny-no-lists-of-lists.vcf.vortex')
