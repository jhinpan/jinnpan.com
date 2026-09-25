#!/usr/bin/env bash
# After the NUMA diagnostic: Step 1A again with SGLang's NUMA binding enabled
# (--cap-add SYS_NICE), then the calibration arms with binding to see which condition
# BBuf's published numbers correspond to.
cd $GB300_WORK
nice="--record-placement --docker-arg=--cap-add=SYS_NICE"
echo "FOLLOWUP_START $(date -u +%FT%TZ)"
ARM_EXTRA="$nice" bash scripts/step1a_current_main.sh \
  n1a-off-a n1a-sim-a n1a-real-a n1a-real-b n1a-sim-b n1a-off-b 2>&1 | tee -a logs/step1a_numa.driver.log
echo "STEP1A_NUMA_RC=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
ARM_EXTRA="$nice" bash scripts/step0_calibration.sh \
  n0-tp4-a n0-ep4-a n0-nods-a n0-tp4-b n0-ep4-b n0-nods-b 2>&1 | tee -a logs/step0_numa.driver.log
echo "STEP0_NUMA_RC=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
echo "FOLLOWUP_DONE $(date -u +%FT%TZ)"
