# Contributing to vLLM-Omni

You may find information about contributing to vLLM-Omni on [Contributing](https://vllm-omni.readthedocs.io/en/latest/contributing/).

Maintainer triage and merge operations are described in the [PR merge loop](docs/contributing/ci/pr_merge_loop.md).

Before code changes or PR submission, check [review and issue lessons](docs/contributing/ci/lessons.md) and apply the entries relevant to your change, verifying their evidence against the current revision.

AI-assisted contributions must follow the [AI Usage Policy](docs/contributing/README.md#ai-usage-policy), including disclosure, human review, and validation requirements.

Local `pre-commit` gates (SPDX, forbidden imports, `torch.cuda`, shellcheck,
mypy, test marks, TTS adapter ratchet, markdownlint, Buildkite schema, and
how to extend allowlists) are documented in the
[Linting](https://vllm-omni.readthedocs.io/en/latest/contributing/#linting)
section of that guide.

Before implementing a contribution, establish the intended behavior and scope with the contributor, reusing requirements already provided. For bug fixes, record a reproducer or other verified failure evidence. Before submitting a PR, complete the [pre-submission checks](docs/contributing/README.md#pre-check-before-submitting), including relevant tests, documentation, and review of the final diff.

vLLM-Omni limits contributors without write access to **5 open PRs**; see the [Pull Request Limit](docs/contributing/README.md#pull-request-limit) section of the contributing guide.
