# Multichallenge Data Directory

The data files in this directory are identical to
`resources_servers/multichallenge_original/data/` — the outsource variant keeps
the same data schema and `agent_ref.name` (`multichallenge_simple_agent`), so
any dataset compatible with the original works here unchanged.

- `example.jsonl` — example tasks (committed) for quick testing.
- `example_metrics.json` / `example_rollouts.jsonl` — example outputs from a
  sample run (committed for reference).
- `advanced/`, `vanilla/`, `advanced.jsonl`, `vanilla.jsonl` — full datasets,
  copied in locally (git-ignored).

See `resources_servers/multichallenge_original/data/README.md` for the full
dataset documentation and regeneration commands.
