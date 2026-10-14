# Private versus non-private single-record replacement experiment

## Question
Does replacing one training record produce a smaller change in a private output than in a non-private constrained score-matching estimator?

## Design
- Fix gamma=0.5, K=40, amplitude=0.25. Use n in {1000,4000,16000}, total alpha in {0.5,2,8}, and 10 independent base datasets per n.
- Freeze the spline basis, bandwidth, truncation radius, and constraints at the alpha=2 model class for every private alpha and the non-private estimator. This makes coefficient vectors directly comparable and prevents bandwidth/dimension changes from masquerading as stability.
- For each dataset create three in-distribution random replacements and two labeled stress replacements. First average replacements within each dataset; inference uses the 10 datasets as independent units.
- Non-private output: constrained empirical score-matching ERM. Private deterministic center: mode of the ridge-regularized private target. Released private output: 100,000-step projected LMC draw.
- For D and D' the private chains start at zero and use identical Gaussian innovations. This synchronous coupling isolates controlled input response, but it is coupling-dependent and is not itself a differential-privacy theorem.
- Separately compare two independent private runs on the same D. This is the mechanism-randomness floor. Report coefficient RMS/max shifts and basis-invariant RMS/max changes of the output score function on 10,000 independent reference points. These are output-shift diagnostics, not clean-score risk.

## Interpretation limits
Differential privacy bounds output distributions for neighboring datasets; it does not require every realized coefficient vector to move less than a non-private ERM under an arbitrary coupling. A smaller private mode shift supports regularization-induced stability. A smaller synchronously coupled LMC shift is useful numerical evidence, while independent-run variability shows whether that response would be visible from one released draw. Stress replacements are diagnostic, not an optimized worst-case search. The private LMC inherits the finite-step mixing/privacy caveat of the main simulations.
