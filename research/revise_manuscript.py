"""Preserve the supplied draft, apply explicit red/green edits, insert results."""
import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manuscript"
text = (M / "original/main.tex").read_text()
edits = []


def replace(old, new, reason):
    global text
    assert text.count(old) == 1, (reason, text.count(old))
    # Keep missing original citation keys visible as editorial text rather
    # than manufacturing a bibliographic source or printing an unresolved '?'.
    display_old = old.replace(r"\cite{nobody}", r"\texttt{[original citation: nobody]}")
    text = text.replace(old, r"\removed{" + display_old + "}" + r"\added{" + new + "}")
    edits.append({"reason": reason, "original": old, "replacement": new})


replace("permtutation", "permutation", "Correct spelling")
replace("accounts for volume conductivity effects by weighting out perfect instantaneous synchrony or asynchrony.",
        "attempts to reduce common-source effects by excluding identical and amplitude-reversed ordinal patterns.",
        "Specify the mask without implying universal artifact removal")
replace("We show that this weighting-out method is mathematically not well-founded, showing that it can lead both to negative mutual information (which should be theoretically impossible) and to potential spurious biases.",
        "This weighted sum need not be nonnegative, because excluding selected contributions removes the nonnegativity guarantee of ordinary mutual information. We give an exact population counterexample and a reproducible EEG illustration.",
        "Distinguish a signed functional from ordinary mutual information")
replace("Using a recently emerging perspective on scaleless information theory, we propose a correction to the weighting procedure for wSMI, creating a new measure, called conditional Symbolic Mutual Information (cSMI).",
        "We propose a correction to the weighting procedure for wSMI, creating a new measure, called weighted Symbolic Divergence (wSD), defined as the retained probability mass multiplied by a KL divergence between the separately normalised retained joint and independent-reference distributions.",
        "Use the requested name and the definition actually supplied")
replace("We show that this new measure behaves similarly to wSMI but is mathematically well-founded, unbiased, and straightforward to implement in current wSMI pipelines.",
        "We prove nonnegativity, establish an upper bound by unweighted symbolic mutual information, and characterise exactly when wSD vanishes despite dependence. An open Python implementation supports deployment in symbolic-connectivity pipelines. The plug-in estimator remains subject to finite-sample bias, and the EEG analysis is an exploratory numerical validation.",
        "Replace unsupported unbiasedness and performance claims with proved and tested claims")
replace(r"accounts for three challenging problems encountered with noisy EEG signals: stability under the presence of noise, non-linear interactions between signals, and accounting for volume conductance effects in EEG \cite{nobody}.",
        r"addresses three challenging problems encountered with noisy EEG signals: robustness to noise, non-linear interactions between signals, and volume conduction effects in EEG \cite{king2013information, imperatori2019eeg}.",
        "Resolve placeholder citation and qualify robustness")
replace(r"Early connectivity measures such as functional connectivity \cite{friston1993functional, friston1994functional, biswal1995functional} or EEG coupling \cite{nobody} look at linear correlations and relationships between the time series of two different signals.",
        r"Early applications of functional connectivity often examined linear correlations and relationships between the time series of two different signals \cite{friston1993functional, friston1994functional, biswal1995functional}.",
        "Functional connectivity is not itself exclusively a linear measure")
replace(r"Information-theoretic connectivity measures capture \textit{all} relationships between signals --- linear and non-linear.",
        r"Mutual information can detect both linear and non-linear dependence between the variables represented, although finite sampling and symbolisation can obscure relationships present in the original signals.",
        "Avoid claiming that coarse ordinal symbols retain every signal dependency")
replace(r"which is invariant to amplitude \cite{bandt2002permutation, king2013information}.",
        r"which is invariant to strictly increasing amplitude transformations in the absence of ties \cite{bandt2002permutation, king2013information}.",
        "State the precise ordinal invariance")
replace(r"the instantaneous conduction of signal, not necessarily representative of actual brain activity or connectivity \cite{nobody}.",
        r"the instantaneous mixing of activity from common sources, which need not represent an interaction between distinct brain regions \cite{imperatori2019eeg}.",
        "Correct the volume-conduction interpretation and resolve citation")
replace("wSD has the useful property that it is strictly positive for all possible distributions, while still accounting for the volume conduction effects that wSMI was designed to handle.",
        "wSD is nonnegative for every joint distribution. It is strictly positive precisely when retained mass is positive and the retained joint distribution is not proportional to the retained product of the original marginals. The measure preserves the exclusions used by wSMI, but a separate validation is needed to establish its sensitivity to genuine interactions and its resistance to volume conduction.",
        "Correct strict positivity and specify the complete equality condition")
replace("behaves locally like mutual information away from the weighted-out volume conduction nodes, and is intuitively familiar to the construction of mutual information as a KL-divergence.",
        "has a nonnegative quadratic expansion near independence, and follows the construction of mutual information as a KL divergence. Dependence confined to excluded cells is appropriately ignored by design. Its complete zero set also includes a uniform proportional change across retained cells, without a change in their relative distribution.",
        "Explain the local behavior and dependence lost by masking")
replace("In our analysis we also demonstrated that this new measure, wSD, is well behaved both on toy systems and experimental data: it is straightforward to calculate and exhibits behaviour familiar to wSMI while strengthening its mathematical foundations.",
        "Our analysis supplies an exact negative-wSMI/positive-wSD example, a signal-level realisation, a finite-sample null study, and an exploratory analysis of public EEG recordings. These results establish the proposed mathematical guarantees and demonstrate implementation on real data; they do not establish clinical usefulness, unbiased estimation, or superiority over wSMI. Appropriate surrogate calibration and independent evaluation remain necessary.",
        "Align the conclusion with the evidence actually generated")

# Production changes to the preamble do not alter scientific prose. The
# untouched preamble is retained in original/main.tex and logged separately.
text = text.replace(r"\documentclass[twocolumns]{IEEEtran}%twoside, lettersize, onecolumn]{IEEEtran}", r"\documentclass[journal]{IEEEtran}")
text = re.sub(r"\\usepackage\[\s*markup=default,\s*authormarkup=none,\s*draft\s*\]\{changes\}", "", text)
text = re.sub(r"\\usepackage\{amsmath, amssymb, tikz, enumitem,.*?tikz-3dplot\}",
              lambda _: r"\usepackage{amsmath,amssymb,tikz,enumitem,bm,amsthm,mathtools,authblk,xcolor,graphicx,booktabs,hyperref}", text)
text = text.replace(r"\usepackage{marginnote}", "")
text = text.replace(r"\showlabelstrue", r"\showlabelsfalse")
text = text.replace(r"\usepackage{subfiles}", "")
macros = r"""
% Review markup: red text is removed; green text is added.
\newif\ifreview
\ifdefined\WSDclean\reviewfalse\else\reviewtrue\fi
\definecolor{addition}{RGB}{0,105,45}
\definecolor{deletion}{RGB}{185,25,25}
\newcommand{\added}[1]{\begingroup\ifreview\color{addition}\fi#1\endgroup}
\newcommand{\removed}[1]{\ifreview{\color{deletion}#1}\space\fi}
\newcommand{\additioncolor}{\ifreview\color{addition}\fi}
\hypersetup{hidelinks}
\let\originalbibitem\bibitem
\renewcommand{\bibitem}[1]{\color{black}\originalbibitem{#1}\ifreview\ifcsname newref@#1\endcsname\color{addition}\fi\fi}
\expandafter\def\csname newref@schalk2009dataset\endcsname{}
\expandafter\def\csname newref@schalk2004bci\endcsname{}
"""
text = text.replace(r"\begin{document}", macros + "\n" + r"\begin{document}")
text = text.replace(r"\maketitle", r"""\maketitle
\ifreview
\begin{center}\small
\textcolor{deletion}{Red: proposed deletion.}\quad
\textcolor{addition}{Green: proposed addition.}
\end{center}
\fi""")
text = text.replace(r"\begin{figure}[h]", r"\added{\input{theory.tex}}" + "\n\n" + r"\begin{figure}[t]")
text = text.replace(r"\begin{tikzpicture}[", r"\resizebox{\columnwidth}{!}{%" + "\n" + r"\begin{tikzpicture}[")
text = text.replace(r"\end{tikzpicture}", r"\end{tikzpicture}}")
old_caption = "The outcomes available from wSMI. The crossed sections indicated are removed in the calculation in order to account for volume conduction."
new_caption = "The six ordinal patterns for kernel length three, labelled by amplitude ranks in time order. Crosses identify identical and amplitude-reversed pairs excluded by both wSMI and wSD. The exclusion is intended to reduce common-source contributions."
text = text.replace(old_caption, r"\removed{"+old_caption+r"}\added{"+new_caption+"}")
text = text.replace(r"\end{figure}", r"\label{fig:mask}"+"\n"+r"\end{figure}", 1)
text = text.replace(r"\section{Validation on sample data}", r"\section{Validation on sample data}"+"\n"+r"\added{\input{validation.tex}}")
text = text.replace(r"\section{Conclusion}", r"\added{\input{implementation.tex}}"+"\n\n"+r"\section{Conclusion}")
(M / "main.tex").write_text(text)
(M / "clean.tex").write_text("\\def\\WSDclean{1}\n\\input{main.tex}\n")
(M / "edit_log.json").write_text(json.dumps({"prose_replacements": edits, "figure_caption": {"original":old_caption,"replacement":new_caption}, "new_sections": ["theory.tex", "validation.tex", "implementation.tex"],
                                           "production_changes": ["Correct IEEEtran class option", "Replace changes package with explicit red/green markup", "Remove unused packages", "Scale original TikZ figure to column width", "Add figure label and working hyperlinks", "Mark newly added bibliography entries green"]}, indent=2)+"\n")

bib = (M / "original/main.bib").read_text()
bib += r"""

@misc{schalk2009dataset,
  author={Schalk, Gerwin},
  title={{EEG Motor Movement/Imagery Dataset}, version 1.0.0},
  howpublished={PhysioNet},
  year={2009},
  note={DOI: 10.13026/C28G6P. Open Data Commons Attribution License v1.0},
  url={https://physionet.org/content/eegmmidb/1.0.0/}
}
@article{schalk2004bci,
  author={Schalk, Gerwin and McFarland, Dennis J. and Hinterberger, Thilo and Birbaumer, Niels and Wolpaw, Jonathan R.},
  title={{BCI2000}: A general-purpose brain-computer interface ({BCI}) system},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={51}, number={6}, pages={1034--1043}, year={2004},
  doi={10.1109/TBME.2004.827072}
}
"""
(M / "main.bib").write_text(bib)
figs = M / "figures"
figs.mkdir(exist_ok=True)
for name in ["synthetic_validation.pdf", "eeg_validation.pdf"]:
    shutil.copy2(ROOT / "results" / name, figs / name)

syn = json.loads((ROOT / "results/synthetic_summary.json").read_text())
eeg = json.loads((ROOT / "results/eeg_summary.json").read_text())
sensitivity = list(csv.DictReader((ROOT / "results/eeg_record_summary.csv").open()))
table_rows = []
for s in range(1, 6):
    rs = [r for r in eeg["records"] if r["record"].startswith(f"S{s:03}")]
    table_rows.append(f"S{s:03} & " + " & ".join(f"{100*r['negative_wsmi']/r['n_pair_epochs']:.1f}" for r in rs) + r" \\")
example = eeg["example"]
validation = r"""\subsection{An exact negative-wSMI counterexample}
To show that negativity is a property of the population functional rather than only an estimation effect, modify the table in Eq.~\eqref{eq:zeroexample} to obtain $P=C/144$, where
\begin{equation}
C=\begin{pmatrix}
10&1&2&0&1&10\\
1&10&0&2&10&1\\
1&1&10&10&1&1\\
1&1&10&10&1&1\\
1&10&1&1&10&1\\
10&1&1&1&1&10
\end{pmatrix}.
\label{eq:counterexample}
\end{equation}
The four modified cells are all retained, and the modification preserves every row and column sum. Therefore $Q_{ij}=1/36$, $m=1/6$ and $z=2/3$ remain unchanged. Within $S$, two cells have count two, two have count zero and twenty have count one. Direct substitution gives
\begin{align}
W&=-\frac{11\log2}{36}\simeq-0.211795,\\
\operatorname{wSD}&=\frac{\log2}{36}\simeq0.019254.
\end{align}
The unweighted symbolic MI is approximately 0.551781 nats. All values are exact population quantities before decimal rounding. Dividing by $\log6$ gives the conventionally normalised measures and preserves the sign difference.

This distribution also has a finite time-series realisation. For each cell $(i,j)$, concatenate $C_{ij}$ paired blocks whose amplitudes equal the three ranks of symbols $i$ and $j$. With kernel three, delay one and stride three, the resulting 432-sample signals give exactly 144 paired symbols and reproduce $C$. Blocks are permuted together with a fixed random seed; no window crosses a block boundary. This construction establishes realisability for disjoint windows, and does not claim that an arbitrary ordinal table is attainable with overlapping windows in a stationary signal.

\subsection{Numerical checks and estimation bias}
We sampled 10,000 joint tables from a symmetric 36-cell Dirichlet distribution with concentration 0.4 per cell (seed 20261019). No table violated nonnegativity or the upper bound by unweighted symbolic MI, and @@NEGATIVE_RANDOM@@ tables gave negative wSMI. The largest discrepancy in Eq.~\eqref{eq:correction} was @@IDENTITY_ERROR@@ nats. These checks test the implementation; the preceding proofs establish the distribution-wide results.

To examine the finite-sample null, we generated 200 independent pairs of independent standard Gaussian sequences at each of five lengths, with kernel three, delay one and overlapping windows. Population wSD is zero in this setting, but its mean plug-in estimate was @@NULL_SHORT@@ nats for 100 samples and @@NULL_LONG@@ nats for 10,000 samples (Fig.~\ref{fig:synthetic}). Thus the proposed rescaling does not make the estimator unbiased. Overlapping windows are also not independent observations, so ordinary independent-cell uncertainty calculations should not be applied without justification.

\begin{figure*}[t]
\additioncolor
\centering
\includegraphics[width=.95\textwidth]{figures/synthetic_validation.pdf}
\caption{Exact and simulated validation. (A) The joint table in Eq.~\eqref{eq:counterexample} gives negative wSMI and positive wSD. (B) Mean estimated wSD for independent Gaussian sequences, using 200 replicates at each length. Its positive finite-sample bias decreases with sample length in this experiment.}
\label{fig:synthetic}
\end{figure*}

\subsection{Exploratory validation on public EEG}
We used the first five participants (S001--S005) and their two baseline recordings, eyes open (R01) and eyes closed (R02), from version 1.0.0 of the PhysioNet EEG Motor Movement/Imagery Dataset \cite{schalk2009dataset, schalk2004bci}. This fixed selection was made before inspecting the connectivity results. The recordings contain 64 EEG channels sampled at 160~Hz. Downloaded files were verified against the provider's SHA256 checksums; exact URLs and hashes are supplied with the analysis code.

Each recording was rereferenced to the common average and filtered continuously with a Butterworth bandpass designed using \texttt{scipy.signal.butter} with $N=4$ (an eighth-order bandpass), applied forward and backward using second-order sections. The lower cutoff was 0.5~Hz; the upper cutoff was $160/(3\tau)$~Hz. The primary delay was $\tau=4$ samples (25~ms; upper cutoff approximately 13.33~Hz); delays one and two were sensitivity analyses. The kernel was three throughout. The symbolic scale and lowpass convention follow the established wSMI methodology \cite{imperatori2019eeg}; this pilot does not reproduce a complete clinical wSMI preprocessing pipeline.

Five non-overlapping 10-second epochs began at 5, 15, 25, 35 and 45 seconds. Thus the first five seconds and all data from 55 seconds onward were excluded. Within each epoch, symbols were computed at every sample start, with no embedding across epoch boundaries. A pair was omitted if either ordinal window contained tied amplitudes; the joint table and both original marginals were computed from the same remaining pairs. There was no pseudocount or bias subtraction. All 2,016 distinct channel pairs were evaluated, giving 100,800 pair-epoch estimates for each delay. No channels or epochs were removed on the basis of connectivity values. There was no manual artifact rejection, ICA or source reconstruction, so artifacts and common-source effects remain possible explanations of individual estimates.

At the primary delay, wSMI was negative in @@EEG_NEGATIVE@@ of 100,800 pair-epochs (@@EEG_PERCENT@@\%), whereas wSD ranged from @@EEG_MIN@@ to @@EEG_MAX@@ nats and never exceeded the corresponding unweighted symbolic MI. Each pair-epoch retained at least @@N_VALID@@ valid paired ordinal windows before the mask. Table~\ref{tab:negative} summarises the descriptive proportions by recording; Fig.~\ref{fig:eeg} shows all primary estimates. These pair-epochs share participants, electrodes and time, and must not be treated as 100,800 independent experimental units.

At delays one and two, respectively, @@TAU_ONE@@\% and @@TAU_TWO@@\% of pair-epochs gave negative wSMI; all wSD estimates again satisfied nonnegativity and the information bound. These proportions are descriptive and depend on the stated preprocessing and symbolic scale.

\begin{table}[t]
\additioncolor
\caption{Pair-epochs with negative wSMI at delay four. Each recording contributes 10,080 pair-epochs.}
\label{tab:negative}
\centering
\begin{tabular}{lrr}
\toprule
Participant & Eyes open (\%) & Eyes closed (\%)\\
\midrule
@@TABLE_ROWS@@
\bottomrule
\end{tabular}
\end{table}

For a reproducible illustration, the most negative primary wSMI occurred in @@EXAMPLE_RECORD@@, channels @@EXAMPLE_X@@--@@EXAMPLE_Y@@, in the epoch beginning at @@EXAMPLE_START@@ seconds. Its wSMI was @@EXAMPLE_WSMI@@ nats, while wSD was @@EXAMPLE_WSD@@ nats, with $m=@@EXAMPLE_M@@$ and $z=@@EXAMPLE_Z@@$. The corresponding integer joint table is supplied as \texttt{eeg\_example\_counts.csv}. This example was selected after analysis to illustrate the sign contrast; it is not an independently selected hypothesis test.

As a descriptive null control, we circularly shifted one symbol sequence within each primary epoch for four prespecified homologous pairs (Fp1--Fp2, C3--C4, P3--P4 and O1--O2), using 99 shifts per pair-epoch, with seed 20261020. Shifts were sampled uniformly with replacement from integer offsets between 320 and $L-320$, where $L$ is the number of ordinal windows. This preserves each symbol sequence's cyclic structure while disrupting their original alignment. Across the 200 pair-epochs, the mean of the surrogate means was @@SHIFT_MEAN@@ nats. The nonzero surrogate baseline reinforces the need for calibration; no significance or diagnostic claims are made. Circular shifts are an approximate control for dependent, potentially nonstationary data, not a guarantee of independence.

\begin{figure*}[t]
\additioncolor
\centering
\includegraphics[width=.95\textwidth]{figures/eeg_validation.pdf}
\caption{Exploratory public-EEG results at kernel three and delay four. (A) All 100,800 pair-epoch estimates, with colour showing logarithmic bin counts and a vertical line at zero wSMI. (B) Descriptive percentages of negative wSMI within each recording. wSD remains nonnegative throughout; this numerical property is not evidence that all retained associations are physiological.}
\label{fig:eeg}
\end{figure*}
"""
values = {
    "NEGATIVE_RANDOM": str(syn["random_audit"]["negative_wsmi"]),
    "IDENTITY_ERROR": "$" + f"{syn['random_audit']['max_identity_error']:.3g}".replace("e-", r"\times10^{-") + "}$",
    "NULL_SHORT": f"{syn['null_summary'][0]['mean_wsd']:.6f}",
    "NULL_LONG": f"{syn['null_summary'][-1]['mean_wsd']:.6f}",
    "EEG_NEGATIVE": f"{eeg['negative_wsmi']:,}",
    "EEG_PERCENT": f"{100*eeg['fraction_negative_wsmi']:.2f}",
    "EEG_MIN": f"{eeg['min_wsd']:.6f}", "EEG_MAX": f"{eeg['max_wsd']:.6f}",
    "N_VALID": str(eeg['min_paired_observations']),
    "TABLE_ROWS": "\n".join(table_rows),
    "EXAMPLE_RECORD": example["record"], "EXAMPLE_X": example["channel_x"], "EXAMPLE_Y": example["channel_y"],
    "EXAMPLE_START": str(example["epoch_start_s"]),
    "EXAMPLE_WSMI": f"{example['wsmi']:.6f}", "EXAMPLE_WSD": f"{example['wsd']:.6f}",
    "EXAMPLE_M": f"{example['retained_mass']:.6f}", "EXAMPLE_Z": f"{example['reference_mass']:.6f}",
    "SHIFT_MEAN": f"{eeg['mean_shift_control_wsd']:.6f}",
    "TAU_ONE": f"{100*sum(int(r['negative_wsmi']) for r in sensitivity if r['delay']=='1')/100800:.2f}",
    "TAU_TWO": f"{100*sum(int(r['negative_wsmi']) for r in sensitivity if r['delay']=='2')/100800:.2f}",
}
for key, value in values.items():
    validation = validation.replace("@@"+key+"@@", value)
assert "@@" not in validation
(M / "validation.tex").write_text(validation)
print("Generated review source, clean wrapper, bibliography, result text and edit log.")
