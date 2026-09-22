# Bridges-2 quickstart

```bash
# 1. Extract and verify
cd ~
tar -xzf nanochat_record_attempt_v1.tar.gz
cd nanochat_record_attempt_v1
./verify_record_package.sh

# 2. Install pinned NanoChat PR #830
bash tools/setup_pr830.sh "$HOME/nanochat-record"

# 3. Persistent data path
export NANOCHAT_BASE_DIR=/path/to/shared/nanochat-record-data
bash tools/prepare_data.sh "$HOME/nanochat-record"

# 4. Configure
cp config/explore.env.example config/explore.env
cp config/confirm.env.example config/confirm.env
cp config/site_bridges2.env.example config/site_bridges2.env
# edit paths and BRIDGES2_ACCOUNT

# 5. Optional 250-step hardware preflight
bash slurm/submit_bridges2_preflight.sh

# 6. Explore
bash slurm/submit_bridges2_explore.sh

# 7. CORE ladder after explore finishes
bash slurm/submit_bridges2_core.sh \
  "$NANOCHAT_BASE_DIR/record_attempts/<RUN_TAG>/bpb_screen/top_candidates.csv"

# 8. Freeze after CORE ladder finishes
python record_tools/freeze_candidate.py \
  --results-dir "$NANOCHAT_BASE_DIR/record_attempts/<RUN_TAG>/core_results" \
  --output config/frozen_candidate.env \
  --threshold 0.256525 --safety-margin 0.0015

# 9. Six confirmation runs
bash slurm/submit_bridges2_confirm.sh
```

Read `README_RECORD_ATTEMPT.md` before spending the full allocation.
