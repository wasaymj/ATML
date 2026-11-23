cd "/kaggle/working/ATML/PA4/Task 3"
/usr/bin/python3 scripts/run_task3.py --step bench --drafts off_policy_s6305,on_policy_s6304 --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED: bench"
/usr/bin/python3 scripts/run_task3.py --step bench_unmerged --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED: bench_unmerged"
