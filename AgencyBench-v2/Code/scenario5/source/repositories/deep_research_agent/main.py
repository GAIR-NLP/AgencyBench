import asyncio
import yaml
import argparse
import json
from tqdm import tqdm
from loguru import logger

from agents.research_agent import ResearchAgent
from agents.browsing_agent import WebBrowsingAgent
from toolkit.search_engine_tool import SearchEngineTool


async def main(args):
    input_data = json.load(open(f"/workspace/data/datasets/{args.split}.json"))

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results = [{"prompt": item["prompt"], "answer": None} for item in input_data]

    output_file = f"/workspace/data/outputs/{args.split}.json"

    for item in tqdm(results):
        web_browsing_agent = WebBrowsingAgent(config)
        search_engine_tool = SearchEngineTool(config)
        research_agent = ResearchAgent(config, web_browsing_agent, search_engine_tool)
        
        try:
            async for answer in research_agent.run(item["prompt"]):
                    item["answer"] = answer
                    break
        except Exception as e:
            logger.error(f"Error in research agent: {e}")
            item["answer"] = "Unable to find the answer"
        
        json.dump(results, open(output_file, "w"), indent=2, ensure_ascii=False)
        logger.info(f"Saved {args.split} outputs to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["dev", "test"])
    args = parser.parse_args()

    asyncio.run(main(args))