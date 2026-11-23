cd "/kaggle/working/ATML/PA4/Task 3"
/usr/bin/python3 scripts/run_task3.py --step validate --artifacts "/kaggle/working/t3_artifacts" && /usr/bin/python3 scripts/run_task3.py --step banks_base --artifacts "/kaggle/working/t3_artifacts" && /usr/bin/python3 scripts/run_task3.py --step quality --models target,original --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED (chain stopped)"
