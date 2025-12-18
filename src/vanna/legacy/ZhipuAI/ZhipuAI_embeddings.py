from typing import List
from zhipuai import ZhipuAI
from chromadb import Documents, EmbeddingFunction, Embeddings
from ..base import VannaBase


class ZhipuAI_Embeddings(VannaBase):
    """
    [future functionality] This function is used to generate embeddings from ZhipuAI.

    Args:
        VannaBase (_type_): _description_
    """

    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if "api_key" not in config:
            raise Exception("Missing api_key in config")
        self.api_key = config["api_key"]
        self.client = ZhipuAI(api_key=self.api_key)

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.client.embeddings.create(
            model="embedding-2",
            input=data,
        )

        return embedding.data[0].embedding


class ZhipuAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A embeddingFunction that uses ZhipuAI to generate embeddings which can use in chromadb.
    usage:
    class MyVanna(ChromaDB_VectorStore, ZhipuAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            ZhipuAI_Chat.__init__(self, config=config)

    config={'api_key': 'xxx'}
    zhipu_embedding_function = ZhipuAIEmbeddingFunction(config=config)
    config = {"api_key": "xxx", "model": "glm-4","path":"xy","embedding_function":zhipu_embedding_function}

    vn = MyVanna(config)

    """

    def __init__(self, config=None):
        if config is None or "api_key" not in config:
            raise ValueError("Missing 'api_key' in config")

        self.api_key = config["api_key"]
        self.model_name = config.get("model_name", "embedding-2")
        self.api_base = config.get("api_base")  # Store api_base if provided

        try:
            # ZhipuAI SDK supports base_url parameter
            init_kwargs = {"api_key": self.api_key}
            if self.api_base:
                init_kwargs["base_url"] = self.api_base
            self.client = ZhipuAI(**init_kwargs)
        except Exception as e:
            raise ValueError(f"Error initializing ZhipuAI client: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        # Replace newlines, which can negatively affect performance.
        input = [t.replace("\n", " ") for t in input]
        all_embeddings = []
        print(f"Generating embeddings for {len(input)} documents")

        # Iterating over each document for individual API calls with retry mechanism
        import time
        max_retries = 3
        retry_delay = 1  # seconds
        
        for idx, document in enumerate(input):
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name, input=document
                    )
                    embedding = response.data[0].embedding
                    all_embeddings.append(embedding)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    # If this is not the last attempt, wait before retrying
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        # Last attempt failed, raise the error
                        raise ValueError(f"Error generating embedding for document after {max_retries} attempts: {e}")

        return all_embeddings
