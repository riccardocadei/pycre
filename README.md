# pycre

Interpretable Discovery and Inference of Heterogeneous Treatment Effects
In health and social sciences, it is critically important to identify subgroups of the study population where a treatment has notable heterogeneity in the causal effects with respect to the average treatment effect (ATE). The bulk of heterogeneous treatment effect (HTE) literature focuses on two major tasks: (i) estimating HTEs by examining the conditional average treatment effect (CATE); (ii) discovering subgroups of a population characterized by HTE.

Several methodologies have been proposed for both tasks, but providing interpretability in the results is still an open challenge. Bargagli-Stoffi et al. (2023) proposed Causal Rule Ensemble, a new method for HTE characterization in terms of decision rules, via an extensive exploration of heterogeneity patterns by an ensemble-of-trees approach, enforcing stability in the discovery. pycre is a Python Package providing a flexible implementation of the Causal Rule Ensemble algorithm.

## Requirents

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```
python scripts/cre.py
```

## References

Causal Rule Ensemble ([methodological paper](https://arxiv.org/abs/2009.09036))
```
@article{bargagli2023causal,
  title={{Causal rule ensemble: Interpretable Discovery and Inference of Heterogeneous Treatment Effects}},
  author={Bargagli-Stoffi, Falco J and Cadei, Riccardo and Lee, Kwonsang and Dominici, Francesca},
  journal={arXiv preprint arXiv:2009.09036},
  year={2023}
}
```