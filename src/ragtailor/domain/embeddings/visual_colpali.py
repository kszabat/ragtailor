from __future__ import annotations

from ragtailor.domain.embeddings.base import VisualEmbedder


class ColPaliVisualEmbedder(VisualEmbedder):
    patch_dimension = 128

    def init(
        self, model_name: str = "vidore/colpali-v1.3", device: str | None = None
    ) -> None:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor
        from colpali_engine.utils.torch_utils import get_torch_device

        self._torch = torch
        self._device = device or get_torch_device(device="auto")
        self._model = ColPali.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device=self._device
        )
        self._processor = ColPaliProcessor.from_pretrained(model_name)

    def embed_pages(self, image_paths: list[str]) -> list[list[list[float]]]:
        from PIL import Image

        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        batch = self._processor.process_images(images).to(self._device)

        with self._torch.inference_mode():
            embeddings = self._model(**batch)

        embeddings_list = embeddings.cpu().to(self._torch.float32).tolist()

        return embeddings_list

    def embed_query(self, text: str) -> list[list[float]]:
        batch = self._processor.process_queries(texts=[text]).to(self._device)

        with self._torch.inference_mode():
            embeddings = self._model(**batch)

        embedding_list = embeddings[0].cpu().to(self._torch.float32).tolist()

        return embedding_list
