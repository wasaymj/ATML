cd "/kaggle/working/ATML/PA4/Task 3"
/usr/bin/python3 scripts/run_task3.py --step bench --drafts original,off_policy_s6304,on_policy_s6305 --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED: bench"
