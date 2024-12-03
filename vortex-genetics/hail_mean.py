import hail as hl


ht = hl.read_table('100_000-no-lists-of-lists.vcf.ht')
ht = ht.select(GT_mean = hl.mean(ht.GT))
ht.write('GT_mean.ht', overwrite=True)
