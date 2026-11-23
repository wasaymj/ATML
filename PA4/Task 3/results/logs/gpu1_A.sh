cd "/kaggle/working/ATML/PA4/Task 3"
/usr/bin/python3 scripts/run_task3.py --step train --artifacts "/kaggle/working/t3_artifacts" && /usr/bin/python3 scripts/run_task3.py --step merge --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED (chain stopped)"
