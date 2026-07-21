import os

import httpx
from chromadb import Documents, Embeddings, EmbeddingFunction

from settings import settings

# chromadb's built-in HuggingFaceEmbeddingFunction still targets the deprecated
# api-inference.huggingface.co/pipeline/feature-extraction/{model} endpoint, which
# 404s -- HF moved feature-extraction behind their router (open issue:
# https://github.com/chroma-core/chroma/issues/2422)
HF_ROUTER_FEATURE_EXTRACTION_URL = "https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"


class HuggingFaceRouterEmbeddingFunction(EmbeddingFunction[Documents]):
    """Calls the HuggingFace Inference API for feature-extraction directly via
    the current router endpoint, since chromadb's built-in HuggingFaceEmbeddingFunction
    still 404s (see the note above HF_ROUTER_FEATURE_EXTRACTION_URL)."""

    def __init__(self, model_name: str, api_key_env_var: str = "HF_TOKEN"):
        """
        Args:
            model_name (str): HuggingFace model id to run feature-extraction with.
            api_key_env_var (str): name (not value) of the env var holding the HF
                token. Stored by name rather than value so get_config()/Chroma's
                persisted collection config never contains the raw secret -- the
                key is re-read from the environment on every reconstruction.
        """
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(f"The {api_key_env_var} environment variable is not set.")
        self.model_name = model_name
        self.api_key_env_var = api_key_env_var
        self._url = HF_ROUTER_FEATURE_EXTRACTION_URL.format(model_name=model_name)
        self._session = httpx.Client(headers={"Authorization": f"Bearer {api_key}"})

    def __call__(self, input: Documents) -> Embeddings:
        """
        Embeds a batch of documents via the HF router feature-extraction endpoint.

        Args:
            input (Documents): texts to embed.

        Returns:
            Embeddings: one embedding vector per input text.
        """
        response = self._session.post(
            self._url, json={"inputs": list(input)}, timeout=60
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def name() -> str:
        """
        Returns:
            str: registry name Chroma stores alongside a collection's persisted config.
        """
        return "huggingface_router"

    def get_config(self) -> dict:
        """
        Returns:
            dict: config Chroma persists to reconstruct this embedding function later.
        """
        return {"model_name": self.model_name, "api_key_env_var": self.api_key_env_var}

    @staticmethod
    def build_from_config(config: dict) -> "HuggingFaceRouterEmbeddingFunction":
        """
        Reconstructs an instance from a previously persisted config.

        Args:
            config (dict): config dict as returned by get_config().

        Returns:
            HuggingFaceRouterEmbeddingFunction: reconstructed instance.
        """
        return HuggingFaceRouterEmbeddingFunction(
            model_name=config["model_name"],
            api_key_env_var=config.get("api_key_env_var", "HF_TOKEN"),
        )

    @staticmethod
    def validate_config(config: dict) -> None:
        """
        Args:
            config (dict): config dict to validate before build_from_config().
        """
        if "model_name" not in config:
            raise ValueError("config must include 'model_name'")
