import hail as hl


ht = hl.read_table('tiny-no-lists-of-lists.vcf.ht')
ht = ht.select(GT_mean = hl.mean(ht.GT))
ht.write('GT_mean.ht', overwrite=True)
