import os
import json
import asyncio
import re
import logging
from typing import Dict, List, Optional, Union, Tuple
from openai import AsyncOpenAI
import random
import httpx

import sys
print(os.getcwd())
print(sys.path)

from code.utils import json_safe_loads
from code.prompt import EXTRACT_CLAIMS_PROMPT, VERIFY_CLAIM_PROMPT, CLAIM_EXTRACTOR_SYSTEM, CLAIM_VERIFIER_SYSTEM, generate_prompt

# Import configuration management
from .config import get_config, Config

class FaithfulnessEvaluator:
    """
    Evaluator for checking the faithfulness of claims in a research response against referenced sources.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration object. If None, will load from default config.
        """
        if config is None: config = get_config()
        
        if not config.validate():
            raise ValueError("Invalid configuration. Please check your environment variables.")
        
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize API client
        client_params = {
            "api_key": config.api_key,
            "base_url": config.base_url
        }
        self.client = AsyncOpenAI(**client_params)
        
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the evaluator"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(self.config.logs_dir, "faithfulness_evaluator.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    ## Phase 1: Claim Extraction

    async def extract_claims_from_response(self, question: str, response: str, question_id: int, response_id: str) -> List[Dict]:
        """
        Extract all claims from a research response that should be verified with or w/o citation.
        
        Args:
            question: The original research question
            response: The response to analyze
            
        Returns:
            List of dictionaries containing claims and their referenced URLs (if available)
        """
        
        # 1.1 Extract references from response
        contents, references = self._extract_contents_and_references(response)
        
        # 1.2 Process response to extract claims
        all_claims = []

        # print("--- Extracting claims from response ---")
        # all_claims = await self._extract_paper_claims(question, contents, references)
        # Extract parts of response
        print("--- Splitting response into sections and extracting references ---")
        sections = self._split_into_sections(contents, step=self.config.claims_step_size)
        
        all_claims = []
        semaphore = asyncio.Semaphore(self.config.max_workers)
        

        async with semaphore:
            # Process each section to extract claims
            print("--- Extracting claims from each section ---")
            # 创建任务列表
            tasks = [
                self._extract_paper_claims(question, section, references)
                for section in sections
            ]

            # 并发执行任务，并收集结果
            results = await asyncio.gather(*tasks)
            for section_claims in results:
                if section_claims: all_claims.extend(section_claims)


        # 1.3 Reformat to group claims with same URL
        # strcuture of each claim: { claim: str, url: str }
        reformated_claims = self._reformat_claims(all_claims)

        # Save claims to a file for debugging
        claims_folder = os.path.join(self.config.claims_dir, f"Q{question_id}")
        os.makedirs(claims_folder, exist_ok=True)

        claims_path = os.path.join(claims_folder, f"{response_id}_claims.json")
        with open(claims_path, 'w', encoding='utf-8') as f:
            json.dump(reformated_claims, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Claims saved to {claims_path}")

        return reformated_claims
    
    def _reformat_claims(self, claims: List[Dict]) -> List[Dict]:
        """
        Reformat and group claims with the same URL.
        
        Args:
            claims: List of claim dictionaries
            
        Returns:
            List of reformatted claim dictionaries
        """
        # Handle edge case of empty claims
        if not claims: return []
        
        # Group claims by URL
        url_to_claims = {}
        for claim in claims:                
            url = claim.get("source", "")
            if url not in url_to_claims:
                url_to_claims[url] = []
            url_to_claims[url].append(claim)
        
        # convert dict into list
        reformatted_claims = []
        for url, claim_list in url_to_claims.items():
            reformatted_claims.append({
                "url": url,
                "claims": claim_list
            })
        
        return reformatted_claims
    
    ## Not used in the current implementation
    def _split_into_sections(self, response: str, step: int = 3) -> List[str]:
        """
        Split the response into sections based on headings.
        
        Args:
            response: The full response text
            
        Returns:
            List of sections
        """
        # List of possible reference section titles
        reference_titles_hash = ['References', 'references', 'Key Citations']
        reference_titles_nohash = ['**References**', '参考资料', '**Sources:**', '**Works cited**', '**引用的著作**', '<div>⁂</div>', '<div style="text-align: center">⁂</div>']

        hash_pattern = '|'.join(re.escape(title) for title in reference_titles_hash)
        nohash_pattern = '|'.join(re.escape(title) for title in reference_titles_nohash)
        reference_pattern = fr'(?i)(?:^#+\s*(?:{hash_pattern})|^#*\s*(?:{nohash_pattern}))'
        
        # Remove references section if it exists
        response = re.split(reference_pattern, response, flags=re.MULTILINE)[0]
        
        # Method 2: Group by every 3 newlines
        sections = re.split(r'\n\n', response)
        sections = [s for s in sections if s.strip() != ""]

        # group every step sections together
        processed_sections = []
        for i in range(0, len(sections), step):
            processed_sections.append("".join(sections[i:i+step]))
        
        # If no sections are found, treat the entire response as one section
        if not processed_sections:
            return [response.strip()]
            
        return processed_sections
    
    def _extract_contents_and_references(self, response: str):
        """
        Extract reference information from the response.
        
        Args:
            response: The full response text
            
        Returns:
            Main content and reference section text or an empty string if not found
        """
        
        # Find the references section if it exists
        reference_titles_hash = ['References', 'references', 'Key Citations']
        reference_titles_nohash = ['**References**', '**Sources:**', '**Works cited**', '**引用的著作**', '<div>⁂</div>', '<div style="text-align: center">⁂</div>']

        hash_pattern = '|'.join(re.escape(title) for title in reference_titles_hash)
        nohash_pattern = '|'.join(re.escape(title) for title in reference_titles_nohash)
        reference_pattern = fr'(?i)(?:^#+\s*(?:{hash_pattern})|^#*\s*(?:{nohash_pattern}))'
        
        try:
            main_match = re.split(reference_pattern, response, flags=re.MULTILINE)[0]
            ref_match = re.split(reference_pattern, response, flags=re.MULTILINE)[1]
        except IndexError:
            ref_match = None

        if not ref_match: 
            return response, ""
        else: 
            return main_match, ref_match
        

    async def _extract_paper_claims(self, question: str, report: str, references: str) -> List[Dict]:
        """
        Use LLM to extract claims from a section that need verification.
        
        Args:
            question: The original research question
            report: The text of the report
            references: Dictionary of references
            section_idx: Index of the section for tracking
            
        Returns:
            List of s dictionaries
        """
        # Create parameters for the prompt
        params = {
            "QUESTION": question,
            "CONTENT": report,
            "REFERENCES": references,
        }
        
        # Generate the prompt using the template
        prompt = generate_prompt(EXTRACT_CLAIMS_PROMPT, params)
        self.logger.debug(f"Extracting claims from sections: {report[:50]}...")

        json_str = ""  # Initialize json_str to avoid UnboundLocalError
        try:
            # Call the API with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.judge_model,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": CLAIM_EXTRACTOR_SYSTEM},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Extract Claims Error (attempt {attempt + 1}): {str(e)}")
                    if attempt == self.config.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Extract and parse JSON from the response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from potential markdown code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', content)
            json_str = content if not json_match else json_match.group(1)
                
            # Clean up any remaining non-JSON content
            json_str = re.sub(r'^[^[{]*', '', json_str)
            json_str = re.sub(r'[^}\]]*$', '', json_str)

            claims = json_safe_loads(json_str)
            return claims
        
        except Exception as e:
            return []
        
    ## Phase 2: Claim Verification


    async def verify_claims(self, claim_list: Dict, semaphore: asyncio.Semaphore) -> List[Dict]:
        """
        Verify a list of claim-paragraph pairs against its referenced URL with rate limiting.
        
        Args:
            claim_list: A dictionary containing url and claim information
            semaphore: Semaphore

        Returns:
            A list of dictionary containing the claim index and its verification result
        """
        async with semaphore:
            url = claim_list.get("url", "")
            claims = claim_list.get("claims", [])

            # Check if the claim has a URL for verification
            if url is None or url == "": 
                self.logger.info("No URL provided for claim.")
                # 给 claims 中的每个 claim 添加一个默认的验证结果
                for claim in claims:
                    claim['result'] = "No URL"
                return claims
            
            try:
                claims_str = "\n".join([f"[{i+1}]\nClaim: {claim['claim']}\nContext: {claim['context']}\n" for i, claim in enumerate(claims)])
                n_claims = len(claims)

                # Attempt to scrape the content from URL
                self.logger.info(f"Verifying claims against URL: {url}")
                content = await self.extract_content_from_url(url)
                
                if not content:
                    self.logger.warning(f"Could not extract content from URL: {url}")
                    for claim in claims:
                        claim['result'] = "URL Error"
                    return claims
                
                result = await self.verify_claims_against_content(n_claims, claims_str, content)
                
                if len(result) != len(claims):
                    self.logger.warning(f"Result length ({len(result)}) does not match claim list length ({len(claims)}). Using fallback.")
                    for claim in claims:
                        claim['result'] = "Unknown"
                    return claims

                # Add the verification result to each claim
                for i, claim in enumerate(claims):
                    claim['result'] = result[i].get('result', 'Unknown')
                
                return claims
                
            except Exception as e:
                self.logger.error(f"Error verifying claims for URL {url}: {str(e)}")
                # 即使出错，也要返回带有错误标记的结果
                for claim in claims:
                    claim['result'] = "Unknown"
                return claims
            

    
    async def extract_content_from_url(self, url: str) -> str:
        """
        Extract content from a given URL using the Jina API.
        Args:
            url: The URL to extract content from
            
        Returns:
            The extracted content as a string
        """
        if not url:
            return ""
        
        if not self.config.jina_api_key:
            self.logger.warning("Jina API key not configured, cannot extract content from URL")
            headers = None
        else:
            headers = { "Authorization": "Bearer " + self.config.jina_api_key }

        jina_url = "https://r.jina.ai/" + url
        async with httpx.AsyncClient() as web_client:
            try:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = await web_client.get(jina_url, headers=headers, timeout=60.0)
                        if response.status_code == 200:
                            content = response.text
                            # Truncate content early to avoid memory issues and token limits
                            if len(content) > self.config.max_content_length:
                                self.logger.info(f"Content from {url} too long ({len(content)} chars), truncating to {self.config.max_content_length} chars")
                                content = content[:self.config.max_content_length] + "\n\n[Content truncated due to length...]"
                            return content
                        else:
                            self.logger.warning(f"HTTP {response.status_code} for URL {url}")
                            if attempt < max_retries - 1:
                                wait_time = 10 + 2 ** attempt + random.uniform(0, 1)
                                self.logger.warning(f"Extract Web Error: Retrying in {wait_time:.2f} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                self.logger.error(f"Failed to extract content from {url} after {max_retries} attempts")
                                return ""
                    except httpx.RequestError as e:
                        self.logger.warning(f"Request error for {url}: {str(e)}")
                        if attempt < max_retries - 1:
                            wait_time = 10 + 2 ** attempt + random.uniform(0, 1)
                            self.logger.warning(f"Extract Web Error: Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            self.logger.error(f"Failed to extract content from {url} after {max_retries} attempts due to request errors")
                            return ""
                    except Exception as e:
                        self.logger.warning(f"Unexpected error for {url}: {str(e)}")
                        if attempt < max_retries - 1:
                            wait_time = 10 + 2 ** attempt + random.uniform(0, 1)
                            self.logger.warning(f"Extract Web Error: Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            self.logger.error(f"Failed to extract content from {url} after {max_retries} attempts due to unexpected errors")
                            return ""
                return ""
            except Exception as e:
                self.logger.error(f"Error extracting content from {url}: {str(e)}")
                return ""

    async def verify_claims_against_content(self, n_claims: int, claims: str, content: str) -> List[Dict]:
        """
        Use LLM to verify a claim against source content.
        
        Args:
            claim: The claim information
            content: The source content from the URL
            url: The URL of the source
            
        Returns:
            String indicating whether the claim is verified or not
        """
        
        # Truncate content if it's too long to avoid token limit
        if len(content) > self.config.max_content_length:
            self.logger.warning(f"Content too long ({len(content)} chars), truncating to {self.config.max_content_length} chars")
            content = content[:self.config.max_content_length] + "\n\n[Content truncated due to length...]"
        
        # Create parameters for the prompt
        params = {
            "SOURCE": content,
            "CLAIM_LIST": claims
        }
        
        # Generate the prompt using the template
        prompt = generate_prompt(VERIFY_CLAIM_PROMPT, params)
        
        try:
            # Call the API with retry logic
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.judge_model,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": CLAIM_VERIFIER_SYSTEM},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    break
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        self.logger.error(f"API call failed after {self.config.max_retries} attempts: {str(e)}")
                        # Return fallback result instead of raising exception
                        return [{"idx": i+1, "result": "Unknown"} for i in range(n_claims)]
                    
                    self.logger.warning(f"Verify Claims Error (attempt {attempt + 1}): {str(e)}")
                    # Retry with exponential backoff
                    timestamp = 2 ** attempt + random.uniform(0, 1) 
                    await asyncio.sleep(timestamp)  # Exponential backoff
            
            # Extract and parse JSON from the response
            content_response = response.choices[0].message.content.strip()

            # Extract JSON from potential markdown code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', content_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content_response
                
            # Clean up any remaining non-JSON content
            json_str = re.sub(r'^[^[{]*', '', json_str)
            json_str = re.sub(r'[^}\]]*$', '', json_str)

            # Avoid Invalid \escape
            json_str = json_str.replace("\\(", "(").replace("\\)", ")")
            
            result = json_safe_loads(json_str)
            return result
        
        except Exception as e:
            self.logger.error(f"Error verifying claim finally: {str(e)}")
            # Return fallback result with error status
            return [{"idx": i+1, "result": "Unknown"} for i in range(n_claims)]
    
    async def evaluate_response_faithfulness(self, question: str, response: str, question_id: int, response_id: str) -> Dict:
        """
        Evaluate the overall faithfulness of a response.
        
        Args:
            question: The original research question
            response: The full response to evaluate
            question_id: ID of the question (int)
            response_id: ID of the response (str)
            
        Returns:
            Dictionary containing faithfulness score and details
        """

        claims_file_path = os.path.join(self.config.claims_dir, f"Q{question_id}", f"{response_id}_claims.json")
        claims = []
        if os.path.exists(claims_file_path):
            with open(claims_file_path, 'r', encoding='utf-8') as f:
                claims = json.load(f)
            if claims != []:
                self.logger.info(f"Claims extracted for Q{question_id}/{response_id}, loading from file.")
        
        if claims == []:
            claims = await self.extract_claims_from_response(question, response, question_id, response_id)

        if not claims:
            return {
                "id": question_id,
                "model": response_id,
                "faithfulness_score": 0.0, 
                "groundedness_score": 0.0,  
                "total_claims": 0,
                "verified_claims": 0,
                "claims_details": []
            }
        
        # Create a semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        # Verify each claim with semaphore for rate limiting
        verification_tasks = [self.verify_claims(claim_list, semaphore) for claim_list in claims]
        all_result = await asyncio.gather(*verification_tasks)
        verified_claims = []
        for result in all_result:
            verified_claims.extend(result)
        
        # Calculate faithfulness score
        t_claim_count = sum(claim["result"].lower() == "yes" for claim in verified_claims)
        f_claim_count = sum(claim["result"].lower() == "no" for claim in verified_claims)
        u_claim_count = sum(claim["result"].lower() == "unknown" for claim in verified_claims)
        n_claim_count = sum(claim["result"].lower() == "no url" for claim in verified_claims)
        
        # Count error cases - these shouldn't be included in verification but should be logged
        error_results = ["error", "api error", "url error", "parse error", "count mismatch"]
        e_claim_count = sum(claim["result"].lower() in error_results for claim in verified_claims)

        verified_claim_count = t_claim_count + f_claim_count
        cited_claim_count    = t_claim_count + f_claim_count + u_claim_count
        total_claim_count    = len(verified_claims)  # Use actual count instead of sum

        faithfulness_score = t_claim_count / verified_claim_count if verified_claim_count else 0.0
        groundedness_score = cited_claim_count / total_claim_count if total_claim_count else 0.0

        # Log statistics for debugging
        self.logger.info(f"Claim statistics for Q{question_id}/{response_id}: "
                        f"Total={total_claim_count}, Yes={t_claim_count}, No={f_claim_count}, "
                        f"Unknown={u_claim_count}, No URL={n_claim_count}, Errors={e_claim_count}")

        if e_claim_count > 0:
            self.logger.warning(f"Found {e_claim_count} claims with errors for Q{question_id}/{response_id}")

        result = {
            "id": question_id,
            "model": response_id,
            "faithfulness_score": faithfulness_score,
            "groundedness_score": groundedness_score,  
            "total_claims": total_claim_count,
            "verified_claims": verified_claim_count,
            "claims_details": verified_claims
        }
            
        return result