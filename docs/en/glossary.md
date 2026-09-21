# Glossary

| Term | Meaning in this repository |
|---|---|
| **AR / plain generation** | Autoregressive generation that commits positions in order and does not revise earlier positions. |
| **Online revision** | Revision interleaved with generation, before the full sequence or late evidence is available. In result directories this is usually `soft` or `hard`. |
| **After-generation revision** | A post-hoc revision pass applied after the initial sequence is generated. Result directories use `posthoc_soft` and `posthoc_hard`. |
| **Hard draft** | A discrete one-hot colour choice for each draft position. |
| **Soft draft / belief state** | A categorical probability vector retained for each draft position instead of immediately taking an argmax. |
| **Temperature** | The configured transformation used in the Belief suite to flatten or sharpen the soft state. Higher values produced more entropy in the tested sweep. |
| **Delayed evidence** | The controlled setup in which early inputs omit relevant edges and the missing evidence is revealed once before the final revision phase. |
| **Whole-question accuracy** | A question is correct only if all six free positions are correct. This is the primary headline metric. |
| **Token / position accuracy** | Per-position accuracy. It can look healthy while whole-question accuracy remains low, so it is not the primary conclusion metric. |
| **Forward-call budget** | Effective model calls per example. It is a more useful comparison axis than raw training steps for several suites. |
| **Post-hoc probe** | A diagnostic that changes the state of already-trained models at inference time; it is not a new training condition. |
| **Official test result** | The serial evaluation stored in `serial_test.json`. For Stage B and Belief, it must not be confused with the validation value in `result.json`. |
| **Smoke / pilot** | Short pipeline checks. They are committed for reproducibility but are not evidence for the research claim. |
| **Blocked** | A run stopped by a resource guard, timeout, interruption, or other supervisor failure. It is not a successful experiment. |
