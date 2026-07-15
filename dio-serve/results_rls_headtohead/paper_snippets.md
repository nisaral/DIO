# NLMS vs RLS vs RR vs LL
mode=sim

- **nlms**: p99=246.7±1.4 frac_fast=1.000±0.000
- **rls**: p99=894.7±13.8 frac_fast=0.403±0.178
- **round_robin**: p99=896.3±9.7 frac_fast=0.500±0.000
- **least_loaded**: p99=246.7±1.4 frac_fast=1.000±0.000

## pick() overhead (µs/request, local)
- nlms: 9.63 µs
- rls: 9.43 µs
- round_robin: 1.67 µs
- least_loaded: 2.18 µs
