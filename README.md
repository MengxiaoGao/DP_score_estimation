# DP_score_estimation

Code for log-scale plots for score estimation error with respect to sample sizes; also for transition phase between non-private and private dominated scenarios.

For rate, use
```
python private_score_experiments_v3.py \
  --experiment rate \
  --quick \
  --gamma 0.5 \
  --target multiscale \
  --rate-reps 10
```

For phase, use
```
python private_score_experiments_v3.py \
  --experiment phase \
  --quick \
  --gamma 0.5 \
  --target multiscale \
  --phase-reps 10
```
