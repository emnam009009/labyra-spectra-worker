# Algorithm Attributions

Labyra implements scientific algorithms inspired by published research. Code in this
repository is **self-written** under Labyra's license — no copyleft (GPL/AGPL/BGMN)
dependencies are included. This document acknowledges the original ideas.

## Multi-phase XRD/Raman matching (R185-4)

**Algorithm**: Greedy iterative peak removal with role-based weight priors.

**Inspired by**:
- Baptista de Castro, P. et al. (2022). "XERUS: An Open-Source Tool for Quick XRD
  Phase Identification and Refinement Automation." *Advanced Theory and Simulations*.
  DOI: 10.1002/adts.202100588.
  - XERUS algorithm: correlation-based similarity → top-N candidates → iterative
    peak removal → refinement.
  - License: GitHub MIT (Xerus-streamlit wrapper); core repo not explicitly licensed
    as of 2026-05.
- Lutterotti, L. et al. (2010). "MAUD: a friendly Java program for material
  analysis using diffraction." *International Union of Crystallography Newsletter* 21.
  - Concept: multi-phase Rietveld refinement.
- Castelli, P. et al. (2024). "Dara: Automated multiple-hypothesis phase
  identification and refinement from powder X-ray diffraction." *arXiv 2510.19667*.
  - Concept: tree search over phase combinations + multi-hypothesis output.

**Implementation**: `src/deviation/multi_phase.py` — original code, no XERUS/MAUD/Dara
binaries or copied source.

## Single-phase Hungarian peak matching (R185-1)

**Algorithm**: Hungarian algorithm (Kuhn 1955) via `scipy.optimize.linear_sum_assignment`.

**Reference**: Kuhn, H. W. (1955). "The Hungarian method for the assignment problem."
*Naval Research Logistics Quarterly* 2(1-2): 83-97.

**License**: scipy is BSD-licensed.

## Physics rules R1-R10 (R185-2)

Each rule cites its primary literature source in the `RuleCitation` field. Examples:

- R1/R2 (strain): Khorsand Zak, A. et al. (2014). DOI: 10.1016/j.solidstatesciences.2014.04.012
- R3 (phonon confinement): Bersani, D. et al. (1998). DOI: 10.1103/PhysRevB.63.125415
- R4 (oxygen vacancy in WO3): Wang, F. et al. (2020). DOI: 10.1021/acs.chemmater.0c02029
- R7 (TMD layer count): Li, H. et al. (2012). DOI: 10.1021/nl201874w
- R8 (amorphization): Tuinstra & Koenig (1970). DOI: 10.1063/1.1674108
- R10 (WS2 resonance): Berkdemir, A. et al. (2013). DOI: 10.1038/srep01755

## Crystal structure & DFT band gap (R184)

**Source**: Materials Project API.

- Jain, A. et al. (2020). "The Materials Project: A materials genome approach to
  accelerating materials innovation." *APL Materials*. DOI: 10.1063/5.0013288.
- License: Materials Project data is CC-BY 4.0; `mp-api` SDK is modified BSD.

## Things Labyra does NOT use

The following are intentionally avoided for license-cleanliness:

| Tool | License | Reason avoided |
|------|---------|----------------|
| Profex | GPL v2 | Copyleft viral incompatible with SaaS |
| BGMN | Proprietary BGMN-only | Not redistributable |
| MAUD | Custom academic license | Unclear commercial terms |
| QualX | Academic only | Not for commercial use |

## Citation requirement

Labyra outputs that use any rule above MUST display the citation chip linking to
the original DOI. This is enforced in the worker `Hypothesis` dataclass:
empty `citation` field → no chip rendered (Trust > Coverage principle).

## Last updated

R185-4 (2026-05-19).
