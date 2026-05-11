"""
MLflow MCP Server — Mammography BI-RADS Experiment Analysis
============================================================
DagsHub üzerindeki MLflow deneylerini Claude Code'a açan MCP server.

Araçlar:
  - list_experiments     : Tüm deneyleri listele
  - search_runs          : Deney içindeki run'ları metriğe göre sırala
  - get_run_details      : Tek bir run'ın tüm metrik ve parametrelerini al
  - compare_runs         : Tüm run'ları karşılaştır (tablo formatı)
  - get_metric_history   : Bir run'da bir metriğin epoch bazlı seyrini al

Çalıştırma (test):
  MLFLOW_TRACKING_URI=... python mlflow_mcp_server.py
"""

import os
import ssl
import urllib3
import mlflow
from mlflow.tracking import MlflowClient
from fastmcp import FastMCP

# SSL / kurumsal ağ kısıtlamaları
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/alilivanturk/Mammography-BIRADS-PNG8Bit.mlflow",
)
DEFAULT_EXPERIMENT = "birads-512-backbone-sweep"

client = MlflowClient(TRACKING_URI)
mcp = FastMCP(name="mlflow-birads")


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _fmt_float(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _get_exp_id(experiment_name: str) -> str | None:
    exps = client.search_experiments(filter_string=f"name = '{experiment_name}'")
    return exps[0].experiment_id if exps else None


# ── Araçlar ──────────────────────────────────────────────────────────────────

@mcp.tool()
def list_experiments() -> str:
    """DagsHub MLflow'daki tüm deneyleri (experiment) listeler."""
    exps = client.search_experiments()
    if not exps:
        return "Hiç deney bulunamadı."
    lines = [f"{'ID':<6} {'Ad'}", "-" * 50]
    for e in exps:
        lines.append(f"{e.experiment_id:<6} {e.name}")
    return "\n".join(lines)


@mcp.tool()
def search_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    max_results: int = 20,
    order_by_metric: str = "val_full_f1_macro",
) -> str:
    """
    Bir deneydeki run'ları belirtilen metriğe göre azalan sırada listeler.

    Parametreler:
        experiment_name  : MLflow deney adı (varsayılan: birads-512-backbone-sweep)
        max_results      : Gösterilecek maksimum run sayısı (varsayılan: 20)
        order_by_metric  : Sıralama metriği (varsayılan: val_full_f1_macro)
    """
    exp_id = _get_exp_id(experiment_name)
    if exp_id is None:
        return f"Deney bulunamadı: '{experiment_name}'"

    runs = client.search_runs(
        [exp_id],
        order_by=[f"metrics.{order_by_metric} DESC"],
        max_results=max_results,
    )
    if not runs:
        return "Bu deneyde hiç run yok."

    lines = [
        f"{'#':<3} {'Run Adı':<45} {order_by_metric:>18}",
        "-" * 70,
    ]
    for i, r in enumerate(runs, 1):
        metric_val = r.data.metrics.get(order_by_metric, float("nan"))
        lines.append(f"{i:<3} {r.info.run_name:<45} {_fmt_float(metric_val):>18}")
    return "\n".join(lines)


@mcp.tool()
def get_run_details(
    run_name: str,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> str:
    """
    Belirli bir run'ın tüm metriklerini ve parametrelerini döner.

    Parametreler:
        run_name         : Run'ın adı (örn: convnextv2_large_seg_v1)
        experiment_name  : MLflow deney adı
    """
    exp_id = _get_exp_id(experiment_name)
    if exp_id is None:
        return f"Deney bulunamadı: '{experiment_name}'"

    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if not runs:
        return f"Run bulunamadı: '{run_name}'"

    r = runs[0]
    lines = [
        f"Run: {r.info.run_name}",
        f"ID:  {r.info.run_id}",
        f"Durum: {r.info.status}",
        "",
        "── METRİKLER ─────────────────────────────────────────",
    ]
    for k, v in sorted(r.data.metrics.items()):
        lines.append(f"  {k:<45} {_fmt_float(v)}")
    lines += ["", "── PARAMETRELER ──────────────────────────────────────"]
    for k, v in sorted(r.data.params.items()):
        lines.append(f"  {k:<45} {v}")
    return "\n".join(lines)


@mcp.tool()
def compare_runs(
    experiment_name: str = DEFAULT_EXPERIMENT,
    max_results: int = 50,
) -> str:
    """
    Bir deneydeki tüm run'ları ana metriklerle karşılaştırır.
    val_full_f1_macro, val_full_auc_roc, test_full_f1_macro, test_full_auc_roc.

    Parametreler:
        experiment_name  : MLflow deney adı
        max_results      : Maksimum run sayısı
    """
    exp_id = _get_exp_id(experiment_name)
    if exp_id is None:
        return f"Deney bulunamadı: '{experiment_name}'"

    runs = client.search_runs(
        [exp_id],
        order_by=["metrics.val_full_f1_macro DESC"],
        max_results=max_results,
    )
    if not runs:
        return "Bu deneyde hiç run yok."

    header = f"{'Run Adı':<45} {'val_f1':>8} {'val_auc':>8} {'tst_f1':>8} {'tst_auc':>8}"
    lines = [header, "-" * 83]
    for r in runs:
        m = r.data.metrics
        vf1  = _fmt_float(m.get("val_full_f1_macro",  float("nan")))
        vauc = _fmt_float(m.get("val_full_auc_roc",   float("nan")))
        tf1  = _fmt_float(m.get("test_full_f1_macro", float("nan")))
        tauc = _fmt_float(m.get("test_full_auc_roc",  float("nan")))
        lines.append(f"{r.info.run_name:<45} {vf1:>8} {vauc:>8} {tf1:>8} {tauc:>8}")
    return "\n".join(lines)


@mcp.tool()
def get_metric_history(
    run_name: str,
    metric_key: str = "val_full_f1_macro",
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> str:
    """
    Bir run'da belirli bir metriğin epoch bazlı tarihçesini döner.

    Parametreler:
        run_name         : Run adı
        metric_key       : İzlenecek metrik (örn: val_full_f1_macro, train_loss)
        experiment_name  : MLflow deney adı
    """
    exp_id = _get_exp_id(experiment_name)
    if exp_id is None:
        return f"Deney bulunamadı: '{experiment_name}'"

    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if not runs:
        return f"Run bulunamadı: '{run_name}'"

    history = client.get_metric_history(runs[0].info.run_id, metric_key)
    if not history:
        return f"'{metric_key}' metriği bu run'da loglanmamış."

    lines = [f"Run: {run_name} — {metric_key}", f"{'Step':>6}  {'Value':>10}"]
    for m in history:
        lines.append(f"{m.step:>6}  {m.value:>10.4f}")
    return "\n".join(lines)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
