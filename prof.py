import pstats
import sys
rank = int(sys.argv[1])
p = pstats.Stats('profile.out')
p.strip_dirs().sort_stats('cumulative').print_stats(rank)
