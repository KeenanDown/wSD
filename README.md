# Weighted Symbolic Divergence

A NumPy implementation of weighted Symbolic Divergence (wSD), with signed
weighted Symbolic Mutual Information (wSMI) and unweighted symbolic MI for
comparison. This repository accompanies the manuscript in `manuscript/`.
The distribution name is `weighted-symbolic-divergence`; the import is `wsd`.
Version 0.1.0 is a local research release, not a claim of PyPI publication.

## Install and use

```sh
python -m pip install .
# For tests and the public EEG replication:
python -m pip install -e '.[test,research]'
```

```python
import numpy as np
from wsd import compare_signals, compare_joint, ordinal_mask, wsd

rng = np.random.default_rng(42)
x = rng.normal(size=5000)
y = 0.5 * x + rng.normal(size=5000)
scores = compare_signals(x, y, order=3, delay=1)
print(scores.wsd, scores.wsmi, scores.smi)
print(scores.retained_mass, scores.reference_mass)
assert 0 <= scores.wsd <= scores.smi + 1e-12

# Exact counterexample: wSMI < 0 and wSD > 0.
counts = np.ones((6, 6), dtype=int)
counts[~ordinal_mask(3)] = 10
counts[0, 2] += 1
counts[1, 3] += 1
counts[0, 3] -= 1
counts[1, 2] -= 1
scores = compare_joint(counts)
assert np.isclose(scores.wsd, np.log(2) / 36)
assert np.isclose(scores.wsmi, -11 * np.log(2) / 36)
```

## Definition and interpretation

Let P be the joint symbol distribution and Q the product of its **original,
unmasked** marginals. For a Boolean retained set S, let m=P(S), z=Q(S).

```text
wSD  = m KL(P(. | S) || Q(. | S))           if m > 0; otherwise 0
wSMI = sum_S P(i,j) log(P(i,j)/Q(i,j))
wSD  = wSMI + m log(z/m)                    if m > 0
0 <= wSD <= SMI
```

wSD is zero exactly when m=0 or P(i,j)=(m/z)Q(i,j) on **every** retained
cell. It can therefore be zero for dependent signals. Identical and
amplitude-inverted signals have m=0 (without ties). A table with count 10
on the diagonal and anti-diagonal and count 1 elsewhere has m=1/6, wSD=0,
and positive ordinary MI. The manuscript proves and explains these cases.
wSD intentionally ignores dependence confined to the masked interactions.
It also vanishes when retained probabilities are a common multiple of the
independence reference, even if m differs from z; the equality condition above
specifies this precisely. Nonzero wSD still depends on m through its prefactor.
It is not a metric or an omnibus independence test. Its plug-in
estimator is not unbiased; positive estimates do not themselves establish
significant coupling or immunity to common-source effects.

## Conventions and API

* Scores default to **nats**, without division by log(order!). Use `base=2`
  for bits, or `normalize=True` to divide all scores by log(alphabet size).
  In normalized mode the log base cancels. The normalized wSD lies in [0,1].
* `patterns(order)` lists zero-based amplitude ranks in **time order**,
  lexicographically. `ordinal_patterns` returns IDs in that alphabet.
  The standard mask removes identical and amplitude-reversed patterns;
  amplitude reversal complements each rank, not the order of time samples.
* `order` is an integer from 2 to 6. Order 2 has no retained cells and
  consequently returns wSD=wSMI=0. Dense joint storage grows as (order!)².
* `delay` is a positive integer in **samples**, not seconds. `stride=1`
  uses overlapping windows. `stride=order` with delay 1 makes disjoint blocks.
* Ties default to an error. `ties='drop'` retains aligned -1 sentinels and
  omits a pair if either window is tied; marginals are then recomputed on
  the valid paired observations. `ties='stable'` breaks ties by temporal
  position. These choices affect estimates and must be reported.
* All inputs must be finite. The package rejects NaNs and infinities.
  There is no implicit jitter, smoothing, pseudocount or truncation of wSMI.
* `compare_symbols(x, y, order=3)` accepts aligned integer symbol IDs, using
  -1 for invalid windows. `compare_joint(table, mask=...)` accepts counts
  or probabilities and an optional arbitrary Boolean mask.
* `compare_signals` returns an immutable `Scores` object; `wsd(x,y,...)`
  returns only its wSD field. A custom asymmetric mask need not yield a
  symmetric measure unless transposed when signals are swapped.
* **Preprocessing is explicit**: apply appropriate EEG filtering, reference,
  artifact handling and epoching externally. Symbolize epochs separately;
  never concatenate them before embedding. No causal interpretation is implied.
* `n_observations` counts valid paired ordinal windows before the mask is
  applied; it is not an effective independent sample size.

The core has only a NumPy dependency. Conditional KL is evaluated with a
cancellation-resistant generalized-KL expression and a series near equal
probabilities, rather than subtracting two potentially large signed terms.
Extremely small probabilities that underflow the reference are rejected.

## Reproduce the analysis

```sh
python -m pytest -q
python research/fetch_eeg.py
python research/run_validation.py
```

The downloader retrieves S001–S005, R01 and R02 from the
[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
and verifies upstream SHA256 values. Raw recordings are ignored by Git.
`data/manifest.json` stores exact URLs, hashes, attribution and selection.
The dataset is distributed under the Open Data Commons Attribution License
v1.0, separately from this repository's GPL-3.0-only software license.
Cite Schalk (2009), DOI 10.13026/C28G6P, and Schalk et al. (2004),
DOI 10.1109/TBME.2004.827072, when using these data.

The fixed pilot uses five 10-second epochs per recording, all 2,016 channel
pairs, order 3 and primary delay 4; delays 1 and 2 are sensitivity analyses.
Filtering, common-average reference, exclusions, surrogate controls, seeds
and limitations are documented in the manuscript and research code.
This is an exploratory mathematical/implementation demonstration, not a
clinical validation study or a test of superiority over wSMI.

Outputs include exact and real-signal joint counts, complete compressed
pair-level results, recording summaries, circular-shift controls, a
finite-sample null study, figures, and environment versions. Channel pairs
and epochs from the same participant are not independent replicates.

## Manuscript

`manuscript/original/` preserves the supplied source byte for byte.
`manuscript/main.tex` is the review version (red deletions, green additions).
`manuscript/clean.tex` accepts the same edits. Compile each from its directory
with Tectonic, or use pdfLaTeX + BibTeX + two pdfLaTeX passes.
`research/revise_manuscript.py` regenerates the edits from the original and
the numerical result files, recording every replacement in `edit_log.json`.

## License

GPL-3.0-only, retaining the repository's existing LICENSE file. The raw EEG
data retain their own attribution license. The manuscript's original text
is supplied by its author; no separate publication license is asserted here.
