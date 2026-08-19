import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import httpx


try:
    import astrbot.api  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    astrbot_module = types.ModuleType("astrbot")
    astrbot_api_module = types.ModuleType("astrbot.api")

    class _Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def exception(self, *_args, **_kwargs):
            pass

    astrbot_api_module.logger = _Logger()
    astrbot_module.api = astrbot_api_module
    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = astrbot_api_module

from utils import dalle_images_api


class DalleImagesApiTests(unittest.IsolatedAsyncioTestCase):
    def test_build_generations_url(self):
        self.assertEqual(
            dalle_images_api._build_generations_url("https://api.example.com"),
            "https://api.example.com/v1/images/generations",
        )
        self.assertEqual(
            dalle_images_api._build_generations_url("https://api.example.com/v1"),
            "https://api.example.com/v1/images/generations",
        )

    def test_extracts_markdown_png_and_structured_url(self):
        png_url = "https://cdn.example.com/output/result.png"
        data = {
            "choices": [
                {
                    "message": {
                        "content": f"生成完成\n![image1]({png_url}) [下载1]\n({png_url})"
                    }
                }
            ],
            "data": [{"url": "https://cdn.example.com/signed?id=1"}],
        }
        candidates = dalle_images_api._extract_image_candidates(data)
        self.assertEqual(candidates.count(png_url), 1)
        self.assertIn("https://cdn.example.com/signed?id=1", candidates)

    def test_filters_gateway_parameters(self):
        filtered = dalle_images_api._filter_dalle_parameters(
            {
                "aspectRatio": "16:9",
                "imageSize": "4K",
                "size": "3840x2160",
                "temperature": 0.7,
            }
        )
        self.assertEqual(
            filtered,
            {"aspect_ratio": "16:9", "image_size": "4K", "size": "3840x2160"},
        )

    async def test_markdown_png_is_downloaded_as_image(self):
        png_url = "https://cdn.example.com/output/result.png"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), png_url)
            return httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=b"\x89PNG\r\n\x1a\nmock-image",
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with patch.object(
                dalle_images_api,
                "_save_image",
                AsyncMock(return_value="images/downloaded.png"),
            ):
                image_url, image_path, message = await dalle_images_api._parse_dalle_response(
                    {"choices": [{"message": {"content": f"![image1]({png_url})"}}]},
                    "",
                    30,
                    client,
                )

        self.assertEqual(image_url, png_url)
        self.assertEqual(image_path, "images/downloaded.png")
        self.assertIsNone(message)

    async def test_text_is_kept_when_no_image_exists(self):
        _, image_path, message = await dalle_images_api._parse_dalle_response(
            {"message": "暂时没有生成图片"},
            "暂时没有生成图片",
            30,
        )
        self.assertIsNone(image_path)
        self.assertEqual(message, "暂时没有生成图片")


if __name__ == "__main__":
    unittest.main()
