# Paths

- **CN:** repository `~/conditioned-iqa/28a_dol/project-1-conditioned-iqa`; datasets `~/conditioned-iqa/data`.
- **VG-Intellect:** datasets `/mnt/hdd1/28d_evs/datasets`.

# CN downloads

- Use `hf_mirror_utils.py` for Hugging Face model snapshots; remote model IDs must not go directly to `from_pretrained`.
- Mirror order is `alpha.hf-mirror.com`, `hf-mirror.com`, then the official Hub; `HF_ENDPOINT` overrides go first and all attempts share the Hub cache.
- Use `download_data_mirrors.py` for IQA archives. Keep local `--weights` offline and preserve resumable downloads when changing endpoints.
