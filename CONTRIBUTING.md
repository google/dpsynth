# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## A Note on the Library's Current State

DP Synth is **rapidly evolving** and currently in a malleable state. We have not
yet made a commitment to backwards compatibility, which means **now is the best
time** to propose API changes, suggest new abstractions, or challenge existing
design decisions. If you have opinions about how the library should be
structured, we want to hear them — once we stabilize the API, the bar for
breaking changes will be much higher.

## Types of Contributions

We welcome contributions in many forms:

### Bug Fixes, Issues, and Documentation

- **Bug reports**: Open an [issue](https://github.com/google/dpsynth/issues)
  describing what went wrong and how to reproduce it.
- **Bug fixes**: Small, focused PRs that fix a specific issue are always
  appreciated.
- **Documentation**: Improving docstrings, adding examples, or clarifying
  existing docs.

### Privacy Hardening

Finding and reporting privacy violations is especially valuable. Contributions
that harden existing implementations — for example, integrating with an
[OpenDP](https://opendp.org/) verification backend as an opt-in check — are
welcome.

### New Features and Mechanisms

We accept new mechanisms at every level of the stack:

- **Tabular data mechanisms**: New algorithms for discrete synthetic data
  generation (e.g., alternatives to AIM, SWIFT, MST).
- **Text generation mechanisms**: New approaches for differentially private text
  synthesis or fine-tuning.
- **Primitives**: New building blocks such as partition selection algorithms or
  numerical discretization strategies.
- **Higher-level components**: Mechanisms that build on top of synthetic data
  generation — for example, adaptive insights from unstructured text, or
  modality-specific pipelines that decompose complex problems into
  subproblems the library already handles well.

### API Design Proposals

The library is at an early stage where API design feedback is highly impactful.
If you think a core abstraction could be improved, open an issue to discuss it
before writing code. We'd rather get the design right now than maintain
backwards-incompatible wrappers later.

### Contributing to Upstream Libraries

DP Synth is co-developed with several companion libraries, and improvements to
any of them directly benefit DP Synth:

- [**MBI**](https://github.com/ryan112358/mbi): Graphical model estimation
  and inference engine used by the tabular mechanisms.
- [**dp_accounting**](https://github.com/google/differential-privacy/tree/main/python/dp_accounting):
  Privacy accounting and budget composition.
- [**jax_privacy**](https://github.com/google-deepmind/jax_privacy):
  Differentially private training primitives used by the text generation
  module.

Contributions to these libraries — bug fixes, performance improvements, new
features — can be surfaced into DP Synth once they land upstream.

## The `contrib/` Directory

If you're proposing a **new mechanism** or a substantial new feature that
doesn't fit neatly into the existing module structure, place it in the
[`contrib/`](contrib/) directory. This is a staging area for experimental or
community-contributed code that hasn't yet been integrated into the core
library.

Once a contribution in `contrib/` has landed, matured, and proven useful, we'll
work with you to migrate it into the appropriate core module.

If your contribution is a bug fix, documentation improvement, or a change to
an existing module, submit it directly to the relevant file — `contrib/` is
only for new standalone additions.

## Acceptance Criteria

When reviewing contributions, we primarily look for two things:

### 1. Simplicity

Simpler contributions with smaller diffs are more likely to be accepted. For
new mechanisms in particular:

- **Aim for a single file** with at most ~500 lines of code.
- You don't need to use the same internal helper functions that existing
  mechanisms use. If your approach is novel, a self-contained implementation
  is perfectly fine.
- However, your mechanism **should conform to the same API contract** as
  existing mechanisms (e.g., implementing `DPMechanism`, accepting the same
  calibration/configuration interface). The exception is mechanisms designed
  for a new data modality (e.g., relational data), where a new API surface
  may be necessary.

### 2. No New Heavy Dependencies

DP Synth deliberately keeps its dependency footprint small:

- **Do not introduce PyTorch, TensorFlow, or other heavyweight
  dependencies.**
- If you need numerical computing, use **JAX**. If you need neural network
  layers, use **Flax**.
- If your contribution requires a new dependency, discuss it in the issue
  tracker first.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to
<https://cla.developers.google.com/> to see your current agreements on file or
to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Before opening a pull request, consider
[filing an issue](https://github.com/google/dpsynth/issues) to discuss the
change. This helps us coordinate efforts and provide guidance on the best
approach.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
