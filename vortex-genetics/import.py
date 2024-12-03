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


def get_ht():
    import hail as hl
    hl.default_reference('GRCh38')

    mt = hl.read_matrix_table('gs://gcp-public-data--gnomad/release/3.1.2/mt/genomes/gnomad.genomes.v3.1.2.hgdp_1kg_subset_dense.mt')
    mt = mt.head(n_rows=100_000)
    mt = mt.select_rows() # remove all row metadata (future work!)
    mt = mt.key_rows_by() # demote row key columns to normal columns
    mt = mt.select_rows(
        # demote locus structure to simple columns for ease of arrow conversion
        chromosome=mt.locus.contig,
        position=mt.locus.position,
    )
    ht = mt.localize_entries('entries') # convert matrix into table with a list of entries per row
    ht = ht.select(
        'chromosome',
        'position',
        # convert list of struct to struct of list
        GT=ht.entries.map(lambda entry: entry.GT.n_alt_alleles()),  # convert from extension data type to a number
        GQ=ht.entries.GQ,
        # # list of lists fields:
        # PL=ht.entries.PL,
        # AD=ht.entries.AD,
    )
    return ht


if not os.path.exists('100_000-no-lists-of-lists.vcf.parquet'):
    print('writing: 100_000-no-lists-of-lists.vcf.parquet')

    df = get_ht().to_pandas()
    df = pa.Table.from_pandas(df)
    pq.write_table(df, '100_000-no-lists-of-lists.vcf.parquet')
else:
    print('found: 100_000-no-lists-of-lists.vcf.parquet')

# if not os.path.exists('100_000-no-lists-of-lists.vcf.ht'):
#     print('writing: 100_000-no-lists-of-lists.vcf.ht')
#     ht = get_ht()
#     ht.write('100_000-no-lists-of-lists.vcf.ht')
# else:
#     print('found: 100_000-no-lists-of-lists.vcf.ht')


if not os.path.exists('100_000-no-lists-of-lists.vcf.vortex'):
    print('writing: 100_000-no-lists-of-lists.vcf.parquet')
    vortex.io.write_path(
        vortex.encoding.compress(
            vortex.array(
                pa.Table.from_pandas(pd.read_parquet('100_000-no-lists-of-lists.vcf.parquet'))
            )
        ),
        '100_000-no-lists-of-lists.vcf.vortex'
    )
else:
    print('found: 100_000-no-lists-of-lists.vcf.parquet')

