"""WTI crude oil analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the energy/oil reference implementation.
"""

from energy_oil_forecasting.analyst_agent.agent import (
    AnchoredForecastOutput,
    AnchoredWtiPromptBuilder,
    WtiPriceForecastPromptBuilder,
    build_wti_agent_predictor,
    build_wti_anchored_config,
    build_wti_basic_config,
    build_wti_code_exec_config,
    build_wti_multitask_news_config,
    build_wti_news_config,
    build_wti_tool_config,
    compress_history,
)
from energy_oil_forecasting.analyst_agent.anchor_lookup import AnchorEntry, AnchorSource
from energy_oil_forecasting.analyst_agent.anchored_predictor import (
    AnchoredAgentPredictor,
    build_wti_anchored_predictor,
)
from energy_oil_forecasting.analyst_agent.news_cache import NewsCacheSource


__all__ = [
    "AnchorEntry",
    "AnchorSource",
    "AnchoredAgentPredictor",
    "AnchoredForecastOutput",
    "AnchoredWtiPromptBuilder",
    "NewsCacheSource",
    "WtiPriceForecastPromptBuilder",
    "build_wti_agent_predictor",
    "build_wti_anchored_config",
    "build_wti_anchored_predictor",
    "build_wti_basic_config",
    "build_wti_code_exec_config",
    "build_wti_multitask_news_config",
    "build_wti_news_config",
    "build_wti_tool_config",
    "compress_history",
]
