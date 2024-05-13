# HNLCalc: Heavy Neutral Lepton Calculator

**By Alec Hewitt and Daniel La Rocco**


## Introduction

We present the package, **HNLCalc**, a fast and fleixble tool which can be used to compute production rates, decay rates, and lifetimes for Heavy Neutral Leptons, a.k.a. Sterile Neutrinos, with arbitrary couplings to the electron, muon, and tau neutrinos. We have implemented over 150 produciton modes and over 100 decay modes, resulting in a comprehensive tool that can be used to study HNL phenomenology for masses up to 10 GeV. 

### Paper

Our primary publication [Simulating Heavy Neutral Leptons with General Couplings at Collider and Fixed
Target Experiments]() provides an overview of the production and decay channels that are implemented in the package. Additionally, it showcases an example of how HNLCalc can be used, where we have used HNLCalc to extend the **FORESEE** (FORward Experiment SEnsitivity Estimator) package to study HNL phenomenology and sensitivity for forward physics experiments. 

### Tutorial

In the repository, we have included a notebook `Example.ipynb`, which outlines the basic usage of HNLCalc and can be used to produce plots for HNL production and decay branching fractions as well as decay length. 

HNL Production Branching Fractions (only a few): 

![Some HNL 3-body Production Branching Fractions](/HNL-3body-Production.png){ width=50% }

HNL Decay Branching Fractions: 
![HNL Decay Branching Fractions](/HNL-Decay.png){ width=50% }

HNL Decay Length: 
![HNL Decay Length](/HNL-ctau.png){ width=50% }

### Support

If you have any questions, please write us at [ahewitt1@uci.edu](ahewitt1@uci.edu) or [laroccod@uci.edu](laroccod@uci.edu).

### References 

- HNL Production: Branching Fractions: [0705.1729](https://arxiv.org/abs/0705.1729)
  - Form factors https://arxiv.org/pdf/hep-ph/0001113
- HNL Decay Branching Fractions: [2007.03701](https://arxiv.org/abs/2007.03701), [1805.08567](https://arxiv.org/abs/1805.08567)
- Hadronic Decay Constants: [1212.3167](https://arxiv.org/abs/1212.3167), [0007169](https://arxiv.org/abs/hep-ph/0007169), [1805.00718](http://arxiv.org/abs/1805.00718), [9907491](http://arxiv.org/abs/hep-ph/9907491), [0508057](http://arxiv.org/abs/hep-ex/0508057), [0610026](http://arxiv.org/abs/hep-ex/0610026), [0705.1729](http://arxiv.org/abs/0705.1729)


