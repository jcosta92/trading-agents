from ..config import AgentSpec
from ..costs import CostModel
from ..registry import Registry
from .base import Agent
from .carry import FundingCarry
from .factor import FactorEquity
from .llmnews import LLMNews
from .simple import BuyAndHold, StaticWeights
from .trend import CryptoTrend, TrendETF
from .vrp import PutCreditSpreads

CLASSES = {c.__name__: c for c in (BuyAndHold, StaticWeights, TrendETF, CryptoTrend, FundingCarry,
                                   FactorEquity, PutCreditSpreads, LLMNews)}


def build_agent(spec: AgentSpec, registry: Registry, cost_model: CostModel) -> Agent:
    return CLASSES[spec.cls](spec, registry, cost_model)
