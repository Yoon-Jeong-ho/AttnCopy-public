# Pre-test implementation corrections

The supervisor's review was applied before any GPU inference. The original
frozen IDs and failed tests remain under `runs/pilot-20261001`. BF16 matches
the original Qwen config; the baseline has no installed Python attention hooks.
Full chat tokenization supplies source character offsets. Individual normalized
function/schema groups overlap 0 times (90 dev, 424 test groups); normalized
queries also overlap 0 times. This does not prove latent language-template
disjointness or absence from Qwen's undisclosed training corpus.

The first GPU smoke produced 0 proposals because the original instruction model
emitted JSON inside Markdown fences on all six smoke inputs. The strict scorer
continues to reject those outputs, consistent with inspected RefCall JSON parsing.
Copy eligibility now accepts a precisely recognized opening ` ```json ` or
unlabelled fence followed by a newline and a JSON call list. The partial JSON
parser still verifies keys, syntax, argument paths and original content offsets;
prose and uncertain wrappers fall back. No prompt, model, head, dataset or strict
accuracy definition changed. The failed smoke is preserved, and an explicit
second smoke attempt must exercise actual draft forwarding before the pilot.

Call-level strict EM sorts calls (order ignored), but never repairs types. Gold
support for events maps unique function identities across ordering; duplicate
same-name calls and unparseable final structures are recorded as ambiguous.
An exactly delimited fenced payload may identify an event's function for this
diagnostic only, without contributing to strict accuracy or JSON validity. Scalars
and malformed outputs stay in the complete example/argument denominator.

During the independent scorer audit, current `refcall_plain/scoring.py` showed
that duplicate-name argument diagnostics use maximum correct typed-leaf
assignment, superseding the historical sequential match inspected initially.
The separate `scoring_review.py` analysis scorer now follows that definition,
tested against exhaustive matching and all duplicate-name order permutations.
Historical cached scores/raw predictions remain unchanged and both argument
aggregates are reported. This correction does not select thresholds or rerun
test inference; call EM, JSON validity and all denominators are unchanged.
