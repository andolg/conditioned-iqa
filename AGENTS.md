# Paths

- **CN:** repository `~/conditioned-iqa/28a_dol/project-1-conditioned-iqa`; datasets `~/conditioned-iqa/data`.
- **VG-Intellect:** datasets `/mnt/hdd1/28d_evs/datasets`.

# CN downloads

- Use `hf_mirror_utils.py` for Hugging Face model snapshots; remote model IDs must not go directly to `from_pretrained`.
- Mirror order is `alpha.hf-mirror.com`, `hf-mirror.com`, then the official Hub; `HF_ENDPOINT` overrides go first and all attempts share the Hub cache.
- Use `download_data_mirrors.py` for IQA archives. Keep local `--weights` offline and preserve resumable downloads when changing endpoints.

# Workflow
- Check if you are running locally or on the `cn-server` machine (`whoami` and `hostname` return `sergey` and `amax`). If local, check the ssh-mcp section below.
- Always use `commit.sh` when committing from the `cn-server` machine to commit without changing global Git config.

# Parallel feature work

- Keep text-conditioning functionality separate from the baseline `train.py`.
  Implement it in `train_text_conditioned.py`, `text_conditioning/`, and
  `configs/text_conditioning/`; do not add text-conditioning flags, models, or
  training branches to `train.py`. Several colleagues and agents develop other
  features concurrently, so the baseline must remain a stable comparison and a
  low-conflict integration point.
- Do not implement learned-label conditioning in the text-conditioning
  workstream. Other colleagues own that feature. Consume their reported results
  as a comparison when available, without copying their implementation into the
  text-conditioning branch.
- Keep learned-query/patch-attention architectures out of the initial
  text-conditioning implementation as well. The project brief treats learned
  queries as a separate research direction. Text conditioning should first keep
  the baseline pooled vision representation fixed; integrate with a colleague's
  stable query scorer later in a separate follow-up if needed.
- Read `docs/text-conditioning-experiments.md` before implementing or running a
  text-conditioning experiment. Preserve its baseline-parity gate, intervention
  controls, MLflow naming, and China/offline Hugging Face requirements.

# ssh-mcp

- Use SSH target `cn-server`. Prefer the purpose-built `ssh-mcp` tools whenever working remotely instead of composing equivalent shell commands.
- Use `ssh_view`, `ssh_glob`, and `ssh_grep` to inspect, find, and search remote files; use `ssh_create` and `ssh_edit` for remote file changes.
- Use `ssh_exec` only for one-off commands that lack a dedicated tool. For interactive or long-running work, reuse a named session with `ssh_ensure_session` and manage it with the list/read/write/stop session tools.
- Use `ssh_scp` or `ssh_sync` for transfers and `ssh_forward` plus the list/stop forward tools for port forwarding.
