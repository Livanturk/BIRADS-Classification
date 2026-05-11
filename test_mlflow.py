from mlflow_mcp_server import list_experiments, search_runs, compare_runs

print("--- EXPERIMENTS ---")
print(list_experiments())

print("\n--- RUNS IN 'birads-1024-8bit-ablation' ---")
print(compare_runs(experiment_name="birads-1024-8bit-ablation"))
