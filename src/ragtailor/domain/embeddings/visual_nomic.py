from __future__ import annotations

from ragtailor.domain.embeddings.base import (
    VisualSingleVectorEmbedder,
    VisualMultiVectorEmbedder,
)


class NomicVisualSingleVectorEmbedder(VisualSingleVectorEmbedder):
    dimension = 3584

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-multimodal-7b",
        device: str | None = None,
    ) -> None:
        import torch
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
        from colpali_engine.utils.torch_utils import get_torch_device

        self._torch = torch
        self._device = get_torch_device(device="auto")
        self._model = BiQwen2_5.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=self._device
        ).eval()
        self._processor = BiQwen2_5_Processor.from_pretrained(model_name)

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        from PIL import Image

        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        batch = self._processor.process_images(images=images).to(self._device)

        with self._torch.inference_mode():
            embeddings = self._model(**batch)

        embeddings_list = embeddings.cpu().to(self._torch.float32).tolist()

        return embeddings_list

    def embed_query(self, query: str) -> list[float]:
        batch = self._processor.process_queries(texts=[query]).to(self._device)

        with self._torch.inference_mode():
            embeddings = self._model(**batch)

        embedding_list = embeddings[0].cpu().to(self._torch.float32).tolist()

        return embedding_list


class NomicVisualMultiVectorEmbedder(VisualMultiVectorEmbedder): ...
