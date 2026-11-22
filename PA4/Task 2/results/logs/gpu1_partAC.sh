cd "/kaggle/working/ATML/PA4/Task 2"
/usr/bin/python3 scripts/run_task2.py --step part_a --cache_dir "/kaggle/working/mqar_cache" --ckpt_dir "/kaggle/working/mqar_checkpoints" || echo "STEP FAILED: part_a"
/usr/bin/python3 scripts/run_task2.py --step part_c --cache_dir "/kaggle/working/mqar_cache" --ckpt_dir "/kaggle/working/mqar_checkpoints" || echo "STEP FAILED: part_c"
