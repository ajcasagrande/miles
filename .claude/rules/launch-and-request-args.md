---
paths:
  - "miles/utils/arguments.py"
  - "miles/utils/eval_config.py"
  - "miles/backends/sglang_utils/**/*.py"
  - "miles/rollout/**/*.py"
  - "miles/router/**/*.py"
  - "miles/tinker/**/*.py"
  - "miles/ray/rollout/**/*.py"
  - "examples/**/*.py"
---

# Launch And Request Args

Follow these rules when adding or changing a CLI flag, a config field (an eval
dataset entry, a `--sglang-config` server group), a per-request field (sampling
params, a `/generate` payload, a session or chat request body), or code that
merges values across those layers. Review the same questions for a change
someone else wrote.

## Pick the scope from what the value means

- **Launch args carry run-wide behavior constraints.** Engine topology and
  features, training/rollout alignment, replay side channels, the served LoRA
  adapter, and server policies are fixed when the run starts. They come from
  `miles/utils/arguments.py`, `--sglang-*`, `--sglang-config` and the
  session-server flags. Validate and derive them once, in `miles_validate_args`
  or `miles/backends/sglang_utils/arguments.py::validate_args`, and treat them
  as read-only afterwards.
- **Request args carry only the control that has to vary.** Sampling knobs,
  the remaining token budget of a resumed sample, per-dataset eval settings, and
  per-turn `chat_template_kwargs` belong to the sample, dataset, turn or client
  call. A launch arg may supply their default.
- **Decide with one question.** Can two requests in the same run correctly use
  different values? If so, the field is a request field with an optional launch
  default. If the run is only correct when every request uses the same value,
  the field is a launch constraint, and request input must not change it.
- **Do not widen either side.** Do not add a launch flag because one call site
  wants a different value; pass it through the request or the dataset config.
  Do not add a request field that lets a caller change a launch constraint.
  Do not choose a value by branching on a model or dataset name in shared code;
  use the layer that owns that identity: TITO model rules, the eval dataset
  entry, or a server group's `overrides`.

## Read the existing precedence before adding a layer

Every value already has a resolver. Read it in the code you are changing;
the chains below describe the resolvers they name.

- **Engine server args** (`_compute_server_args` in
  `miles/backends/sglang_utils/sglang_engine.py`): server-group `overrides` >
  values miles derives from its own args (`tp_size` from
  `--rollout-num-gpus-per-engine`, memory saver, replay, LoRA, metrics, ...) >
  `--sglang-<field>` > SGLang default. A `--sglang-<field>` only fills a field
  miles has not derived, so setting it for a derived field does nothing.
  `validate_args` also rewrites some `--sglang-*` values, including `tp_size`,
  deterministic inference and router policy.
- **Eval engines** (`_compute_eval_raw_model` in `sglang_config.py`): YAML group
  `overrides` > `--eval-sglang-*` > eval defaults (replay outputs off; dp, pp,
  ep and attention cp sizes reset to 1 when eval tp differs) > `--sglang-*`.
- **Eval sampling** (`build_eval_dataset_configs` in
  `miles/utils/eval_config.py`): dataset entry > eval `defaults` > `--eval-*` >
  `--rollout-*`.
- **Session chat requests** (`prepare_chat_request` in
  `miles/rollout/session/request_args.py`): server constraints from launch
  config come first. They force TITO fields, follow the replay flags, select the
  served adapter, and return HTTP 400 for conflicting control fields. Session
  sampling defaults then fill only omitted fields, and a training temperature
  must match. TITO model rules follow, with `chat_template_kwargs` resolved as
  request > continued turn > `--apply-chat-template-kwargs`. Evaluation finally
  turns replay outputs off.
- **`/generate` rollouts**: `compute_sampling_params` builds the base from
  launch args, `generate_and_rm` hands each request a `deepcopy`, and
  per-request adjustments such as the partial-rollout budget, seeds and custom
  generate functions apply to that copy.

## Change a value inside its chain

- **Extend the owning resolver.** Add the value at the layer that matches its
  scope. Do not create a parallel path or a second knob for the same field, for
  example an `--eval-*` flag outside `DATASET_RUNTIME_SPECS`, or a generate
  function that reads `args.rollout_temperature` instead of the
  `sampling_params` it receives.
- **Lower layers fill or narrow; they never silently contradict.** When a
  request sets a value that launch config owns, reject it with a clear error,
  as the session server does, instead of overwriting or forwarding it. Force a
  field silently only when miles must own it for the protocol, and document it
  as forced (`logprobs` and `return_meta_info` on the session path).
- **Keep derived values from hiding user flags.** When miles derives or forces a
  field, reject the conflicting user flag during validation, or document at the
  flag that miles replaces it, as the CLI reference does for the eval fleet's
  `tp_size`.
- **Every accepted field must reach its consumer.** A flag, config key or
  request field that parses but is never read turns a user setting into a
  silent no-op. Wire it through the resolver or refuse it.
- **Keep train and eval differences in the resolver.** When evaluation relaxes a
  launch constraint, such as an overridable temperature or disabled replay
  outputs, implement that in the shared resolver rather than at call sites.
- **Never mutate launch state per request.** Do not write to `args`, a shared
  base `sampling_params`, or launch kwargs from a request path; copy first.

## Verify the resolution

For each added or changed field, the change description must answer:

1. **Scope:** is it launch or request, and which answer does the decision
   question give?
2. **Layer:** which resolver owns it, and where does it sit in that chain?
3. **Conflict:** when two layers disagree, does the higher layer win, fill, or
   reject, and can the user observe the result?
4. **Reach:** does the value arrive through every entry path it claims (train,
   eval, session, `/generate`, tinker)? Assert on the resolved artifact, such
   as the rendered SGLang argv or the resolved request body. Existing coverage
   lives in `tests/fast/backends/sglang_utils/test_compute_server_args.py`,
   `tests/fast/backends/sglang_utils/test_sglang_config.py` and
   `tests/fast/rollout/session/test_request_args.py`.
5. **Docs:** when user-visible precedence changes, update
   `docs/user-guide/cli-reference.md` or the guide that documents it, such as
   `docs/user-guide/agentic-rollout.md`.
