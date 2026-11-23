cd "/kaggle/working/ATML/PA4/Task 3"
/usr/bin/python3 scripts/run_task3.py --step banks_students --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED: banks_students"
/usr/bin/python3 scripts/run_task3.py --step align --artifacts "/kaggle/working/t3_artifacts" || echo "STEP FAILED: align"
