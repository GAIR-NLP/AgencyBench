from typing import Any, Dict, AsyncGenerator, List, Optional
import os
import asyncio
import aiohttp
import json
import re
from loguru import logger
from collections import defaultdict

from numpy import result_type
from transformers import AutoTokenizer

from services.llm_service import LLMService


WEB_BROWSING_PROMPT = """You are a helpful assistant that can extract and summarize the relevant information for the given search query from the content of the given URLs.

This is the search query:
<search_query>
{search_query}
</search_query>

This is the content of the URL:
<webpage>
{webpage}
</webpage>

Please check carefully whether there is relevant information for the search query in the content of the URL. If there is, please extract the relevant information and then summarize it in a few sentences and wrap it with <useful_information> tag. If there is no relevant information for the search query in the content of the URL, please put an empty string (i.e. <useful_information></useful_information>). Thus, your output should be in the following format:

<useful_information>
( Put the extracted useful information for the search query here )
</useful_information>
"""


def get_content_from_tag(
    content: str,
    tag: str,
    default_value: Optional[str] = None
) -> Optional[str]:
    """
    extract the content from the first specified tag
    """
    if not content:
        return default_value

    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value


class WebBrowsingAgent:
    """Tool for performing web searches using Google via Serper API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.serper_api_key = config["serper_api_key"]
        self.llm_service = LLMService(
            model_name=config["web_browsing_model"]["name"],
            api_key=config["web_browsing_model"]["api_key"],
            base_url=config["web_browsing_model"]["base_url"],
            provider=config["web_browsing_model"]["provider"],
            api_version=config["web_browsing_model"]["api_version"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained("toolkit/qwen-2.5-72b-instruct-tokenizer")

        self.max_chunk_num_per_webpage = 5
        self.max_chunk_size = 4000
        self.overlap_size = 100


    async def run(self, webpages: List[Dict[str, Any]], research_question: str):
        """Process URLs and extract relevant information.
        
        Args:
            webpages: The list of webpages to browse
            research_question: The research question to extract the useful information from the content
        """

        # get the webpage content
        successfully_scraped_webpages = await self._get_webpage_content(webpages)
        if not successfully_scraped_webpages:
            return None


        async def get_single_completion(message):
            try:
                # get the completion from the llm service
                result = self.llm_service.get_chat_completion_sync(
                    messages=[{"role": "user", "content": message["prompt"]}],
                    temperature=0.6,
                    max_tokens=2000
                )
                
                if result is None:
                    return None
                
                useful_information = get_content_from_tag(result.choices[0].message.content, "useful_information")

                if not useful_information or len(useful_information) < 5:
                    return None

                return {
                    "useful_information": useful_information,
                    "url": message["url"],
                    "query": message["query"]
                }
                
            except Exception as e:
                logger.error(f"single request error: {str(e)}")
                return None

        messages = []
        for webpage in successfully_scraped_webpages:
            for chunk in webpage['content_chunks']:
                messages.append({
                    "url": webpage["url"],
                    "query": webpage["query"],
                    "prompt": WEB_BROWSING_PROMPT.format(
                        webpage=chunk,
                        search_query=webpage["query"]
                    )
                })
        
        logger.info(f"number of browsing chunks: {len(messages)}")

        # process the requests in batches, avoid sending too many requests at once
        batch_size = 20  # the number of requests to process in each batch
        info_extraction_results = []
        num_of_messages_processed = 0
        
        # use semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(batch_size)  # the maximum number of concurrent requests
        
        async def process_with_semaphore(message):
            async with semaphore:
                return await get_single_completion(message)
    

        # process the requests in batches
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i+batch_size]
            logger.info(f"process batch {i//batch_size + 1}/{(len(messages)+batch_size-1)//batch_size}, contains {len(batch_messages)} requests")

            # create the tasks for the current batch
            batch_tasks = [process_with_semaphore(message) for message in batch_messages]
            # execute the requests in the current batch concurrently
            batch_returns = await asyncio.gather(*batch_tasks)
            info_extraction_results.extend(batch_returns)

            num_of_messages_processed += len(batch_messages)
        
        info_extraction_results = [result for result in info_extraction_results if result is not None]

        logger.info(f"number of responses: {len(info_extraction_results)}")

        url_to_useful_contents = defaultdict(list)
        for ret in info_extraction_results:
            url_to_useful_contents[ret["url"]].append(ret["useful_information"])

        results = {}
        for url, useful_contents in url_to_useful_contents.items():
            results[url] = "\n\n".join(useful_contents)

        return results
        
    async def _get_webpage_content(self, webpages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        successfully_scraped_pages = []
        
        async def scrape_task(webpage):
            content_chunks = await self._scrape_webpage_content_with_serper(webpage["url"])
            return {
                "url": webpage["url"],
                "query": webpage["query"],
                "content_chunks": content_chunks
            }

        tasks = [scrape_task(webpage) for webpage in webpages]
        
        for future in asyncio.as_completed(tasks):
            scraped_page = await future
            if scraped_page["content_chunks"]:
                successfully_scraped_pages.append(scraped_page)

        return successfully_scraped_pages
    
    async def _scrape_webpage_content_with_serper(self, url: str) -> str:
        is_pdf = self._is_pdf_url(url)
        
        payload = json.dumps({
            "url": url,
            "includeMarkdown": True
        })
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://scrape.serper.dev/", data=payload, headers=headers) as response:
                    response.raise_for_status()  # Will raise an exception for 4xx/5xx statuses
                    data_json = await response.json()

            webpage_content = ""
            if (is_pdf or "markdown" not in data_json) and "text" in data_json:
                webpage_content = data_json["text"]
            elif "markdown" in data_json:
                webpage_content = data_json["markdown"]

            tokens = self.tokenizer.tokenize(webpage_content)
            if len(tokens) < 10:
                return None
            

            # Split tokens into chunks with 100 token overlap
            chunk_size = self.max_chunk_size  # Adjust this value based on your needs
            overlap_size = self.overlap_size

            chunks = []
            
            if len(tokens) > chunk_size:
                for i in range(0, len(tokens), chunk_size - overlap_size):
                    chunk = tokens[i:i + chunk_size]
                    if len(chunk) > 10:  # Only add non-empty chunks
                        chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
            else:
                # If content is small enough, process normally
                chunks.append(webpage_content)
            return chunks[:self.max_chunk_num_per_webpage]
            
        except aiohttp.ClientError as e:
            logger.error(f"Aiohttp error when scraping {url}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing page {url} with Serper: {str(e)}")
        
        return None
    
    def _is_pdf_url(self, url: str) -> bool:
        """
        check if the url is a pdf url
        
        Args:
            url: the url to check
            
        Returns:
            bool: if the url is a pdf url, return True, otherwise return False
        """
        import re
        from urllib.parse import urlparse, unquote


        if not url or not isinstance(url, str):
            return False
        
        try:
            # parse the url
            parsed_url = urlparse(url.lower())
            # decode the url
            path = unquote(parsed_url.path)
            domain = parsed_url.netloc
            
            # check if the url is a pdf url
            if path.endswith('.pdf'):
                return True
            
            # check if the url is an arxiv pdf url
            if 'arxiv.org' in domain:
                arxiv_patterns = [
                    r'/pdf/\d{4}\.\d{4,5}(\.pdf)?$',
                    r'/pdf/[a-z-]+/\d{7}(\.pdf)?$',
                    r'/pdf/[a-z-]+\.\w+/\d{7}(\.pdf)?$'
                ]
                for pattern in arxiv_patterns:
                    if re.search(pattern, path):
                        return True
            
            # check if the url is a pdf url from other common pdf services
            pdf_domains_patterns = {
                'drive.google.com': r'/file/d/[^/]+/',
                'dropbox.com': r'\.pdf(\?|$)',
                'onedrive.live.com': r'\.pdf(\?|$)',
                'researchgate.net': r'\.pdf(\?|$)',
                'academia.edu': r'\.pdf(\?|$)',
                'mendeley.com': r'\.pdf(\?|$)',
                'semanticscholar.org': r'/paper/.*\.pdf$',
                'biorxiv.org': r'\.full\.pdf$',
                'medrxiv.org': r'\.full\.pdf$',
            }
            
            for pdf_domain, pattern in pdf_domains_patterns.items():
                if pdf_domain in domain and re.search(pattern, path):
                    return True
            
            # check if the url is a pdf url
            if '/' in path:
                filename = path.split('/')[-1]
                if filename.endswith('.pdf'):
                    return True
            
            # check
            query = parsed_url.query.lower()
            if query and any(param in query for param in ['format=pdf', 'type=pdf', 'download=pdf']):
                return True
                
            return False
            
        except Exception as e:
            # if the url parsing fails, fall back to simple string check
            logger.warning(f"URL parsing failed, using simple check: {e}")
            url_lower = url.lower()
            return url_lower.split('?')[0].split('#')[0].endswith('.pdf')