from typing import Any, Dict, List, Union, AsyncGenerator
import os
import asyncio
import aiohttp
import json
from loguru import logger

class SearchEngineTool:
    """Tool for performing web searches using Google via Serper API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["serper_api_key"]
        self.api_url = "https://google.serper.dev/search"

        self.max_retries = 3
        self.timeout = 10
        self.top_k = 10
    
    async def _search_single(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a web search for a single query.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        retries = 0
        while retries < self.max_retries:
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        'X-API-KEY': self.api_key,
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        "q": query,
                        "num": self.top_k,
                        "gl": "cn",
                        "type": "search",
                        "engine": "google"
                    }
                    
                    logger.info(f"Searching for query: {query}")
                    async with session.post(self.api_url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Search API request failed with status {response.status}")
                            retries += 1
                            await asyncio.sleep(1 * retries)  # 指数退避
                            continue
                        
                        response_data = await response.json()
                        
                        # Extract relevant information from search results
                        results = []
                        if "organic" in response_data:
                            for item in response_data["organic"][:self.top_k]:
                                result = {
                                    "url": item.get("link", ""),
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "query": query  # Include the original query
                                }
                                results.append(result)
                        
                        logger.info(f"Found {len(results)} results for query: {query}")
                        return results
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout searching for query '{query}', retry {retries+1}/{self.max_retries}")
                retries += 1
                await asyncio.sleep(1 * retries)  # 指数退避
                continue
            except Exception as e:
                logger.error(f"Error performing search for query '{query}': {str(e)}")
                retries += 1
                await asyncio.sleep(1 * retries)  # 指数退避
                continue
                
        logger.error(f"Failed to search for query '{query}' after {self.max_retries} retries")
        return []  # 返回空列表而不是None，避免下游处理错误
    
    async def search(self, queries: Union[str, List[str]]) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Perform web searches using the specified queries.
        
        Args:
            queries: A single search query or a list of search queries
            
        Returns:
            List of search results, each containing url, title, and snippet
        """
        # Handle single query case
        if isinstance(queries, str):
            return await self._search_single(queries)
        
        # Handle multiple queries in parallel
        search_tasks = [self._search_single(query) for query in queries]
        search_results = await asyncio.gather(*search_tasks)
        
        # Deduplicate and combine results from all queries using URL as key
        url_to_result = {}
        for results in search_results:
            if results:
                for result in results:
                    url = result.get("url")
                    if url and url not in url_to_result:
                        url_to_result[url] = result
        
        combined_results = list(url_to_result.values())
        
        logger.info(f'Number of search results for {queries} after deduplication: {len(combined_results)}')
        
        return combined_results