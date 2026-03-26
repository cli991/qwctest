from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union, Tuple
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import requests
import json as json_lib
import time
import re
import torch


def simple_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


JSON_ONLY_SYSTEM_PROMPT = (
    "You must return ONLY a valid JSON object. "
    "Do not include markdown fences, explanations, or any extra text."
)


def clean_json_text(text: str) -> str:
    """Clean common wrappers around model JSON output."""
    if text is None:
        return ""

    text = str(text).strip()

    # remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # keep only the outermost JSON object if extra text exists
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start:end + 1]

    return text.strip()



def safe_json_loads(text: str, default: Optional[dict] = None) -> dict:
    """Safely parse model JSON output and return a default on failure."""
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response repr: {repr(text)}")
        if default is None:
            default = {}
        return default


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        """Get completion from LLM"""
        pass


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content


class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(
                    prop_schema["type"],
                    prop_schema.get("items")
                )

        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ollama completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)


class SGLangController(BaseLLMController):
    def __init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(
                    prop_schema["type"],
                    prop_schema.get("items")
                )

        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            json_schema = response_format.get("json_schema", {}).get("schema", {})

            payload = {
                "text": JSON_ONLY_SYSTEM_PROMPT + "\n\n" + prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 300,
                    "json_schema": json_schema
                }
            }

            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")

            result = response.json()
            text = result.get("text", "")
            text = clean_json_text(text)

            # 如果清洗后仍然不是合法 JSON，返回空模板
            try:
                json.loads(text)
                return text
            except Exception:
                print(f"SGLang raw response repr: {repr(result.get('text', ''))}")
                empty_response = self._generate_empty_response(response_format)
                return json.dumps(empty_response)

        except Exception as e:
            print(f"SGLang completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LiteLLMController(BaseLLMController):
    """LiteLLM controller for universal LLM access including Ollama and SGLang"""
    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"]["schema"]
        result = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(
                    prop_schema["type"],
                    prop_schema.get("items")
                )

        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "response_format": response_format,
                "temperature": temperature
            }

            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key

            response = completion(**completion_args)
            return response.choices[0].message.content

        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)


class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            ollama_model = f"ollama/{model}" if not model.startswith("ollama/") else model
            self.llm = LiteLLMController(
                model=ollama_model,
                api_base="http://localhost:11434",
                api_key="EMPTY"
            )
        elif backend == "sglang":
            self.llm = SGLangController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', or 'sglang'")


class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):

        self.content = content

        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            print("analysis", analysis)
            keywords = keywords or analysis.get("keywords", [])
            context = context or analysis.get("context", "General")
            category = category or analysis.get("category", "Uncategorized")
            tags = tags or analysis.get("tags", [])

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:
        """Analyze content to extract keywords, context, and other metadata"""
        prompt = """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags
4. Choosing a short category label

Return ONLY a valid JSON object in the following format:
{
    "keywords": [],
    "context": "",
    "category": "",
    "tags": []
}

Content for analysis:
""" + content

        default_analysis = {
            "keywords": [],
            "context": "General",
            "category": "Uncategorized",
            "tags": []
        }

        response = ""
        try:
            response = llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "context": {
                                "type": "string"
                            },
                            "category": {
                                "type": "string"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                        },
                        "required": ["keywords", "context", "category", "tags"],
                        "additionalProperties": False
                    },
                    "strict": True
                }}
            )

            analysis = safe_json_loads(response, default_analysis)
            if not isinstance(analysis, dict):
                return default_analysis

            analysis.setdefault("keywords", [])
            analysis.setdefault("context", "General")
            analysis.setdefault("category", "Uncategorized")
            analysis.setdefault("tags", [])
            return analysis

        except Exception as e:
            print(f"Error analyzing content: {e}")
            print(f"Raw response repr: {repr(response)}")
            return default_analysis


class HybridRetriever:
    """Hybrid retrieval system combining BM25 and semantic search."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def _rebuild_bm25(self):
        tokenized_docs = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)

        state = {
            'alpha': self.alpha,
            'bm25': self.bm25,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': 'all-MiniLM-L6-v2'
        }

        try:
            state['model_name'] = self.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            pass

        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)

        retriever = cls(model_name=state['model_name'], alpha=state['alpha'])
        retriever.bm25 = state['bm25']
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})

        embeddings_path = Path(retriever_cache_embeddings_file)
        if embeddings_path.exists():
            retriever.embeddings = np.load(embeddings_path)

        return retriever

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, alpha: float):
        all_docs = [", ".join(m.keywords) for m in memories.values()]
        retriever = cls(model_name, alpha)
        retriever.add_documents(all_docs)
        return retriever

    def add_documents(self, documents: List[str]) -> bool:
        if not documents:
            return False

        self.corpus = list(documents)
        self.document_ids = {doc: idx for idx, doc in enumerate(self.corpus)}
        self._rebuild_bm25()
        self.embeddings = self.model.encode(documents)
        return True

    def add_document(self, document: str) -> bool:
        if document in self.document_ids:
            return False

        self.corpus.append(document)
        self.document_ids[document] = len(self.corpus) - 1
        self._rebuild_bm25()

        doc_embedding = self.model.encode([document])
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embedding])

        return True

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus:
            return []

        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query)) if self.bm25 is not None else np.zeros(len(self.corpus))

        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]

        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores

        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()


class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def add_documents(self, documents: List[str]):
        if not documents:
            return

        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus:
            return []

        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)

        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")

        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")

        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")

        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever


class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval"""
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        self.memories = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base, sglang_host, sglang_port)
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return ONLY a valid JSON object with the following structure:
                                {{
                                    "should_evolve": true,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": [0, 1],
                                    "tags_to_update": ["tag_1", "tag_2"],
                                    "new_context_neighborhood": ["new context 1", "new context 2"],
                                    "new_tags_neighborhood": [["tag_1"], ["tag_2"]]
                                }}
                                '''
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)
        ])
        if evo_label is True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self):
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)

        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_memory = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices)
        )
        print("prompt_memory", prompt_memory)

        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            response_format={"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "should_evolve": {
                            "type": "boolean",
                        },
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "suggested_connections": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        },
                        "new_context_neighborhood": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "tags_to_update": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "new_tags_neighborhood": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "required": [
                        "should_evolve",
                        "actions",
                        "suggested_connections",
                        "tags_to_update",
                        "new_context_neighborhood",
                        "new_tags_neighborhood"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }}
        )

        print("response", response, type(response))
        default_response = {
            "should_evolve": False,
            "actions": [],
            "suggested_connections": [],
            "tags_to_update": note.tags if note.tags else [],
            "new_context_neighborhood": [],
            "new_tags_neighborhood": []
        }
        response_json = safe_json_loads(response, default_response)
        print("response_json", response_json, type(response_json))

        should_evolve = response_json.get("should_evolve", False)
        if should_evolve:
            actions = response_json.get("actions", [])
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json.get("suggested_connections", [])
                    new_tags = response_json.get("tags_to_update", note.tags if note.tags else [])
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json.get("new_context_neighborhood", [])
                    new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve, note

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str


def run_tests():
    print("Starting Memory System Tests...")

    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openai',
        llm_model='gpt-4o-mini'
    )

    print("\nAdding test memories...")

    memory_ids = []
    memory_ids.append(memory_system.add_note(
        "Neural networks are composed of layers of neurons that process information."
    ))

    memory_ids.append(memory_system.add_note(
        "Data preprocessing involves cleaning and transforming raw data for model training."
    ))

    print("\nQuerying for related memories...")
    query = MemoryNote(
        content="How do neural networks process data?",
        llm_controller=memory_system.llm_controller
    )

    related_text, related_indices = memory_system.find_related_memories(query.content, k=2)
    print("related_text", related_text)
    print("related_indices", related_indices)


if __name__ == "__main__":
    run_tests()
