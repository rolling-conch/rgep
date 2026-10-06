# Source attribution

This repository contains original research extensions and dependencies adapted from the CSAM-ResUNet project at https://github.com/V1S10NAL/CSAM-ResUNet.

Upstream reference: Huang, Weixiang; Chen, Jiajin; Xiong, Hao; Shao, Ligang; Wang, Guishi; Liu, Kun; Chen, Chilai; Gao, Xiaoming. *Cascaded Improved Neural Network for the Reconstruction, Classification, and Unmixing of the Raman Spectra of Mixed Microplastics*. Analytical Chemistry (2026), DOI: 10.1021/acs.analchem.5c04049.

Upstream model files retain their existing contents. This publication bundle does not claim authorship of those original architectures and does not grant a replacement license for upstream work. No standalone upstream LICENSE file was present in the inspected checkout; public redistribution permissions should be confirmed before a public release.

`models/RSSNetInspired.py` and `models/ReferenceGuidedExtractor.py` are task-specific inspired/adapted implementations. Cite the corresponding RSSNet and reference-guided extraction work when using them; they are not official reproductions from those authors.

The dataset and trained weights are distributed separately by their providers and are not included here. The fixed split manifest contains row indices and checksums only. Historical experiments and manuscript drafts are excluded.

Release-only adaptation: the historical-results test in `tests/test_reference_refresh_ablation.py` explicitly skips when separately obtained result/manuscript fixtures are absent. `SOURCE_MANIFEST.json` records the original source hashes before this adaptation. No model or training algorithm is changed by this packaging step.

