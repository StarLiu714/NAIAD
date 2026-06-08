# Third-Party Licenses

This repository contains NAIAD code together with code adapted from or
compatible with external projects. Each project's license governs its portion
of this repository.

## NAIAD (Our code)
- License: GNU General Public License v3.0 (GPLv3)
- See: `LICENSE`

## NA-MPNN
- Upstream: https://github.com/baker-laboratory/NA-MPNN
- License: MIT
- License text: https://github.com/baker-laboratory/NA-MPNN/blob/main/LICENSE
- Usage: NAIAD is based on and extends the original NA-MPNN training and
  inference code.

## NA-MPNN BSD-licensed utility files
The upstream NA-MPNN license file identifies the following files as covered by
the BSD 3-Clause License:

- `cifutils.py`
- `obutils.py`
- `geometry.py`

Original notice from upstream:

- Copyright (c) 2025 University of Washington.
- Developed at the Institute for Protein Design by Ivan Anishchenko, Indrek
  Kalvet and Rohith Krishna.
- License: BSD 3-Clause
- Source license text: https://github.com/baker-laboratory/NA-MPNN/blob/main/LICENSE

## pdbx
`pdbx` is pinned as a git submodule in this repository.
- Upstream: https://github.com/soedinglab/pdbx/tree/master
- License: no license file is provided in the upstream repository.
- Usage: optional dependency used by the mmCIF parsing utilities.
- Note: because the upstream repository does not provide an explicit license,
  users should consult the upstream project or authors before redistributing
  pdbx itself. NAIAD does not vendor pdbx in this repository.
