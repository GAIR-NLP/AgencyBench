import json
from typing import Dict, Any, List
from loguru import logger

from services.llm_service import LLMService
from agents.research_agent_prompts import DECISION_MAKING_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, FINAL_ANSWER_PROMPT_TEMPLATE
from agents.browsing_agent import WebBrowsingAgent
from utils import get_content_from_tag, get_all_content_from_tag
from toolkit.search_engine_tool import SearchEngineTool


class ResearchAgent:
    def __init__(
            self, 
            config: Dict[str, Any],
            web_browsing_agent: WebBrowsingAgent,
            search_engine_tool: SearchEngineTool
    ):
        self.llm_service = LLMService(
            model_name=config["agent_model"]["name"],
            api_key=config["agent_model"]["api_key"],
            base_url=config["agent_model"]["base_url"],
            provider=config["agent_model"]["provider"],
            api_version=config["agent_model"]["api_version"],
        )
        self.web_browsing_agent = web_browsing_agent
        self.search_engine_tool = search_engine_tool
        
        # Add anything if you want

    async def run(self, research_question: str):
        # Add anything if you want
        pass
    
    async def _search_and_browse(self, research_question: str, search_queries: List[str]):
        pass
        