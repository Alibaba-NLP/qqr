"""
Custom CLI argument registration for RTI (Rollout-Time Intervention) extensions.

Called via slime's `parse_args(add_custom_arguments=add_rti_arguments)`.
"""


def add_rti_arguments(parser):
    def _add_if_missing(option, *args, **kwargs):
        # ``--save-rollout-data`` may also be registered by a slime patch
        # (e.g. qqr/patches/slime-rg-sed.patch); skip to avoid a conflict.
        if any(option in action.option_strings for action in parser._actions):
            return
        parser.add_argument(option, *args, **kwargs)

    # ── Rollout extensions ──
    _add_if_missing(
        "--save-rollout-data", action="store_true", default=False,
        help="Save rollout samples as jsonl files.",
    )
    parser.add_argument(
        "--rollout-max-turns", type=int, default=None,
        help="Maximum number of turns for agent rollout.",
    )
    parser.add_argument(
        "--eval-max-turns", type=int, default=None,
        help="Maximum number of turns for eval rollout (falls back to rollout-max-turns).",
    )
    parser.add_argument(
        "--custom-group-rollout-function-path", type=str, default=None,
        help="Python path to custom generate_and_rm_group function.",
    )

    # ── Inspector (RTI) ──
    parser.add_argument("--inspector-enabled", action="store_true", default=False)
    parser.add_argument("--inspector-ratio", type=float, default=1.0)
    parser.add_argument("--inspector-log-dir", type=str, default=None)
    parser.add_argument("--inspector-use-judge-llm", action="store_true", default=False)
    parser.add_argument("--inspector-openai-base-url", type=str, default=None)
    parser.add_argument("--inspector-openai-api-key", type=str, default="EMPTY")
    parser.add_argument("--inspector-openai-model", type=str, default=None)
    parser.add_argument("--inspector-temperature", type=float, default=0.0)
    parser.add_argument("--inspector-max-new-tokens", type=int, default=800)
    parser.add_argument("--inspector-timeout", type=int, default=300)
    parser.add_argument("--inspector-fallback-enabled", action="store_true", default=True)
    parser.add_argument("--inspector-replace-num", type=int, default=0)
    parser.add_argument("--custom-inspector-prompt-path", type=str, default=None)
    parser.add_argument("--custom-inspector-method-path", type=str, default=None)
    parser.add_argument("--enable-postcheck", action="store_true", default=False)
    parser.add_argument("--inspector-allow-gt-bootstrap", action="store_true", default=False)
    parser.add_argument("--inspector-strict-direction-match", action="store_true", default=False)

    # ── Ref model / KL gate ──
    parser.add_argument("--calc-ref-prob-diff", action="store_true", default=False)
    parser.add_argument("--enable-ref-kl-gate", action="store_true", default=False)
    parser.add_argument("--ref-judge-method", type=str, default="ppl")
    parser.add_argument("--ref-ppl-threshold", type=float, default=5.0)
    parser.add_argument("--ref-js-threshold", type=float, default=0.3)
    parser.add_argument("--ref-kl-threshold", type=float, default=1.0)
    parser.add_argument("--ref-kl-temperature", type=float, default=1.0)
    parser.add_argument("--ref-kl-min-prefix-tokens", type=int, default=0)
    parser.add_argument("--save-inspector-data", action="store_true", default=False)

    # ── User simulator (for tau2_bench etc.) ──
    parser.add_argument("--user-sim-openai-base-url", type=str, default=None)
    parser.add_argument("--user-sim-openai-api-key", type=str, default="EMPTY")
    parser.add_argument("--user-sim-model", type=str, default=None)
    parser.add_argument("--user-sim-temperature", type=float, default=0.0)
    parser.add_argument("--user-sim-max-new-tokens", type=int, default=1024)
    parser.add_argument("--user-sim-timeout", type=int, default=180)
    parser.add_argument("--user-sim-max-retries", type=int, default=3)
    parser.add_argument("--user-sim-retry-base-backoff", type=float, default=10.0)
    parser.add_argument("--user-sim-retry-max-backoff", type=float, default=60.0)

    # ── Step-level length penalty (tau2_bench group_reward) ──
    parser.add_argument(
        "--step-length-penalty-soft-tokens", type=int, default=0,
        help="Per-step response length (tokens) below which no penalty applies. "
             "0 (default) disables the penalty entirely.",
    )
    parser.add_argument(
        "--step-length-penalty-cap", type=float, default=1.0,
        help="Max per-step penalty subtracted from the step reward, "
             "reached at rollout-max-response-len.",
    )

    # ── Judge LLM ──
    parser.add_argument("--judge-openai-base-url", type=str, default=None)
    parser.add_argument("--judge-openai-api-key", type=str, default="EMPTY")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-timeout", type=int, default=60)
    parser.add_argument("--judge-think", action="store_true", default=True)
    parser.add_argument("--no-judge-think", action="store_false", dest="judge_think",
                        help="Disable thinking mode for the judge LLM")

    # ── ClawBench ──
    parser.add_argument("--clawbench-max-completion-tokens", type=int, default=9096,
                        help="Max completion tokens for ClawBench agent LLM calls.")
    parser.add_argument("--clawbench-max-steps", type=int, default=50,
                        help="Max trajectory steps for ClawBench rollout.")
    parser.add_argument("--clawbench-sandbox-timeout", type=int, default=2400,
                        help="E2B sandbox lifetime in seconds for ClawBench.")
    parser.add_argument("--clawbench-harness", type=str, default="openclaw",
                        help="Agent harness for ClawBench sandbox (browser-use, openclaw).")
    parser.add_argument("--synthetic-task-upload", action="store_true", default=False,
                        help="Upload synthetic task data into the E2B sandbox before running.")
    parser.add_argument("--synthetic-task-base-dir", type=str, default=None,
                        help="Base directory for resolving synthetic task directories by task name.")
    parser.add_argument("--max-idle-nudges-train", type=int, default=0,
                        help="Max nudges when agent stops without tool call during TRAINING rollouts.")
    parser.add_argument("--max-idle-nudges-eval", type=int, default=3,
                        help="Max nudges when agent stops without tool call during EVAL rollouts.")

    # ── ClawBench-Harness RTI (reuses inspector:* args for endpoint/replace/temp) ──
    parser.add_argument("--rti-prompts-dir", type=str, default=None,
                        help="Directory containing judge/corrector/postcheck .txt prompts for RTI logger.")

    return parser
