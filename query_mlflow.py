import os
import ssl
import urllib3
from mlflow.tracking import MlflowClient

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

client = MlflowClient('https://dagshub.com/alilivanturk/Mammography-BIRADS-PNG8Bit.mlflow')

runs = client.search_runs(['19'])

for r in runs:
    print(f"Run Name: {r.data.tags.get('mlflow.runName', 'Unknown')}")
    print(f"Status: {r.info.status}")
    print("Metrics:")
    metrics = {k: v for k, v in r.data.metrics.items() if 'val' in k or 'test' in k or 'train' in k}
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    print("Params:")
    params = {k: v for k, v in r.data.params.items() if k in ['backbone', 'model.use_lateral_fusion', 'model.use_bilateral_fusion', 'model.use_binary_head', 'model.use_subgroup_head', 'loss.focal_gamma']}
    for k, v in sorted(params.items()):
        print(f"  {k}: {v}")
    print("-" * 50)
