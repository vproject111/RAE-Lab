from fastapi import FastAPI
from typing import List
from hypothesis_engine import HypothesisEngine
from experiment_orchestrator import ExperimentOrchestrator
from strategy_compiler import StrategyCompiler
from safe_rollout_manager import SafeRolloutManager
from rae_core.models.failure import FailureLearningRecord

app = FastAPI(title="RAE-Lab Evolution Engine v3.1")
hyp_engine = HypothesisEngine()
orch = ExperimentOrchestrator()
compiler = StrategyCompiler()
rollout = SafeRolloutManager()

@app.get("/health")
async def health():
    return {"status": "fully_operational", "department": "lab"}

@app.post("/lab/experiment/run")
async def run_experiment(proposal_id: str):
    return orch.run_offline_replay(proposal_id, [])

@app.post("/lab/analyze/failures")
async def analyze_failures(failures: List[FailureLearningRecord]):
    return hyp_engine.generate_from_failures(failures)

@app.get("/v2/lab/insights")
async def get_lab_insights(project: str = "RAE-Suite", limit: int = 10):
    from metrics_aggregator import get_lab_insights_endpoint
    return await get_lab_insights_endpoint()

@app.post("/api/scan")
async def register_scan(report: dict):
    from metrics_aggregator import register_scan_endpoint
    return await register_scan_endpoint(report)

@app.get("/api/mab/weights")
async def get_mab_weights():
    from metrics_aggregator import get_mab_weights_endpoint
    return await get_mab_weights_endpoint()

@app.get("/api/guardrails/candidates")
async def get_candidate_guardrails(status: str = "CANDIDATE"):
    from metrics_aggregator import get_candidate_guardrails_endpoint
    return await get_candidate_guardrails_endpoint(status=status)


