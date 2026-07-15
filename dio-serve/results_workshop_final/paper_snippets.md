# Workshop final suite — paper snippets
Generated: 2026-07-15T17:00:01.414765+00:00
Model: Qwen/Qwen2.5-3B-Instruct
Tokenizer: Qwen/Qwen2.5-3B-Instruct
Status: ok

## W2 MAPE tokenizer vs heuristic
- **heuristic**: MAPE $89.7\pm4.9$ (n=3)
- **tokenizer**: MAPE $117.5\pm10.9$ (n=3)
- delta (tok − heur) pp: 27.72

## W3 Regime A n=10 (homogeneous dual-GPU)
- **nlms**: p99 $3413.1\pm30.5$ (ci95 $\pm18.9$, n=10); MAPE $122.8\pm5.9$; frac_e0 $0.447\pm0.039$; vs RR $-2.6\pm1.0\%$
- **rls**: p99 $3466.5\pm10.8$ (ci95 $\pm6.7$, n=10); MAPE $132.4\pm6.0$; frac_e0 $0.120\pm0.023$; vs RR $-4.2\pm0.6\%$
- **round_robin**: p99 $3326.3\pm17.1$ (ci95 $\pm10.6$, n=10); MAPE $87.9\pm2.4$; frac_e0 $0.500\pm0.000$
- **least_loaded**: p99 $3312.7\pm429.7$ (ci95 $\pm266.3$, n=10); MAPE $141.0\pm12.6$; frac_e0 $1.000\pm0.000$; vs RR $0.4\pm13.1\%$

## W4 Regime C real throttle n=10
- **nlms**: p99 $3427.1\pm38.4$ (ci95 $\pm23.8$, n=10); MAPE $126.2\pm12.3$; frac_e0 $0.470\pm0.076$; vs RR $48.3\pm0.7\%$
- **rls**: p99 $5087.1\pm1504.8$ (ci95 $\pm932.7$, n=10); MAPE $119.0\pm9.9$; frac_e0 $0.367\pm0.189$; vs RR $23.2\pm22.9\%$
- **round_robin**: p99 $6631.6\pm36.7$ (ci95 $\pm22.7$, n=10); MAPE $102.3\pm2.7$; frac_e0 $0.500\pm0.000$

## W5 Longer decode max_tokens=128
- **nlms**: p99 $13835.0\pm73.2$ (ci95 $\pm82.8$, n=3); MAPE $127.2\pm6.9$; frac_e0 $0.550\pm0.150$; vs RR $-0.1\pm0.9\%$
- **round_robin**: p99 $13820.9\pm87.7$ (ci95 $\pm99.2$, n=3); MAPE $97.2\pm8.1$; frac_e0 $0.500\pm0.000$

## W6 Coefficient ±50% (live)
- max |Δp99| vs baseline: 50.56489265407055
- tier_p50: 49.42%
- tier_m50: 47.97%
- cache_p50: 47.94%
- cache_m50: 48.48%
- both_p50: 50.56%
- both_m50: 49.98%

## Artifacts
- Full JSON: `/kaggle/working/results_workshop_final/summary.json`
- Logs: `/kaggle/working/results_workshop_final/logs`
