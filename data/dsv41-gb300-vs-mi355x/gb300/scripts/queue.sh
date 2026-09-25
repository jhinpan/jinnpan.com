#!/usr/bin/env bash
# GPU queue run inside tmux so it survives the agent session. Each driver stops at its
# first failed arm; a failed calibration does not block Step 1A.
cd $GB300_WORK
echo "QUEUE_START $(date -u +%FT%TZ)"
bash scripts/step0_calibration.sh s0-nods-a s0-tp4-b s0-ep4-b s0-nods-b 2>&1 | tee -a logs/step0.driver.log
echo "STEP0_RC=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
bash scripts/step1a_current_main.sh s1a-off-a s1a-sim-a s1a-real-a s1a-real-b s1a-sim-b s1a-off-b 2>&1 | tee -a logs/step1a.driver.log
echo "STEP1A_RC=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
echo "QUEUE_DONE $(date -u +%FT%TZ)"
