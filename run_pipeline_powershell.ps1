$ErrorActionPreference = "Stop"
$RUN_NAME = "final_run_core_context_6"

python -u scripts/00_audit_raw.py --config config/config.yaml --run-name $RUN_NAME
python -u scripts/01_preprocess_daily.py --config config/config.yaml --run-name $RUN_NAME
python -u scripts/02_build_summary.py --config config/config.yaml --run-name $RUN_NAME
python -u scripts/03_run_clustering.py --config config/config.yaml --run-name $RUN_NAME
python -u scripts/04_generate_visuals.py --config config/config.yaml --run-name $RUN_NAME
python -u scripts/05_generate_report.py --config config/config.yaml --run-name $RUN_NAME

Write-Host "Pipeline selesai. Output ringkas: outputs/runs/$RUN_NAME"
