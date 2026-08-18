import asyncio
import base64
import binascii
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from astrbot.api import logger


_MAX_MESSAGE_CHARS = 1200


def _truncate_message(text: str, max_chars: int = _MAX_MESSAGE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(message truncated)"


def _build_images_url(api_base: str, operation: str) -> str:
    """Build an OpenAI-compatible Images API URL from either a root or /v1 base."""
    base = (api_base or "https://api.openai.com").rstrip("/")
    if base.endswith("/v1") or base.endswith("/v1beta/openai"):
        return f"{base}/images/{operation}"
    return f"{base}/v1/images/{operation}"


def _decode_input_image(value: str) -> Tuple[bytes, str, str]:
    mime_type = "image/png"
    encoded = value or ""
    if encoded.startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
            match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64$", header)
            if match:
                mime_type = match.group(1)
        except ValueError:
            pass

    normalized = re.sub(r"\s+", "", encoded)
    try:
        raw = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 reference image: {exc}") from exc

    suffix_map = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/png": "png",
    }
    return raw, mime_type, suffix_map.get(mime_type.lower(), "png")


async def _save_image_bytes(content: bytes, suffix: str = "png") -> str:
    plugin_root = Path(__file__).parent.parent
    images_dir = plugin_root / "images"
    images_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    file_path = images_dir / f"gpt_image_{ts}_{uid}.{suffix}"
    file_path.write_bytes(content)
    return str(file_path)


def _output_suffix(data: Dict[str, Any]) -> str:
    output_format = str(data.get("output_format") or data.get("format") or "png").lower()
    if output_format in ("jpeg", "jpg"):
        return "jpg"
    if output_format == "webp":
        return "webp"
    return "png"


def _detect_output_suffix(content: bytes, fallback: str = "png") -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WEBP":
        return "webp"
    return fallback


async def _parse_image_response(
    data: Dict[str, Any], timeout_seconds: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            error = error.get("message") or error
        return None, None, _truncate_message(str(error or "API 响应未包含图片数据"))

    first = items[0] if isinstance(items[0], dict) else {}
    b64_json = first.get("b64_json")
    if b64_json:
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", str(b64_json)), validate=True)
            suffix = _detect_output_suffix(raw, _output_suffix(first))
            image_path = await _save_image_bytes(raw, suffix)
            return None, image_path, first.get("revised_prompt")
        except (binascii.Error, ValueError) as exc:
            return None, None, f"API 返回了无效的 base64 图片：{exc}"

    image_url = first.get("url")
    if image_url:
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.get(str(image_url))
                response.raise_for_status()
            content_type = response.headers.get("content-type", "image/png").lower()
            suffix = "jpg" if "jpeg" in content_type else "webp" if "webp" in content_type else "png"
            image_path = await _save_image_bytes(response.content, suffix)
            return str(image_url), image_path, first.get("revised_prompt")
        except Exception as exc:
            return str(image_url), None, f"下载生成图片失败：{exc}"

    return None, None, "API 响应的 data[0] 中没有 b64_json 或 url"


def _filter_image_parameters(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Only forward supported parameters; GPT image quality defaults to high."""
    aliases = {
        "imageSize": "size",
        "image_size": "size",
        "outputFormat": "output_format",
        "outputCompression": "output_compression",
        "moderation": "moderation",
        "quality": "quality",
        "size": "size",
        "output_format": "output_format",
        "output_compression": "output_compression",
        "background": "background",
        "n": "n",
    }
    result: Dict[str, Any] = {}
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            mapped = aliases.get(key)
            if mapped and value is not None:
                result[mapped] = value
    # Apply to both /images/generations and /images/edits. Explicit user input wins.
    result.setdefault("quality", "high")
    return result


async def generate_or_edit_image_openai(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 300,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Generate or edit an image through an OpenAI-compatible Images API."""
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    keys = [str(key) for key in api_keys if str(key).strip()]
    if not keys:
        return None, None, "未提供 GPT API Key"

    reference_images = input_images_b64 or []
    operation = "edits" if reference_images else "generations"
    url = _build_images_url(api_base, operation)
    parameters = _filter_image_parameters(generation_config)
    last_message: Optional[str] = None

    for api_key in keys:
        headers = {"Authorization": f"Bearer {api_key}"}
        for attempt in range(max(1, max_retry_attempts)):
            if attempt:
                await asyncio.sleep(min(2 ** attempt, 10))
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    if reference_images:
                        form_data: Dict[str, str] = {"model": model, "prompt": prompt}
                        form_data.update({key: str(value) for key, value in parameters.items()})
                        files = []
                        for index, encoded in enumerate(reference_images):
                            raw, mime_type, suffix = _decode_input_image(encoded)
                            files.append(("image[]", (f"reference_{index}.{suffix}", raw, mime_type)))
                        response = await client.post(url, headers=headers, data=form_data, files=files)
                    else:
                        payload: Dict[str, Any] = {"model": model, "prompt": prompt}
                        payload.update(parameters)
                        response = await client.post(
                            url,
                            headers={**headers, "Content-Type": "application/json"},
                            json=payload,
                        )

                if response.status_code == 429 or response.status_code >= 500:
                    last_message = _truncate_message(response.text)
                    logger.warning(f"GPT Images API 暂时失败 {response.status_code}，准备重试")
                    continue
                if response.status_code < 200 or response.status_code >= 300:
                    try:
                        error_data = response.json()
                        error = error_data.get("error", error_data)
                        if isinstance(error, dict):
                            error = error.get("message") or error
                        last_message = _truncate_message(str(error))
                    except Exception:
                        last_message = _truncate_message(response.text)
                    return None, None, last_message

                return await _parse_image_response(response.json(), timeout_seconds)
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_message = _truncate_message(str(exc))
                logger.warning(f"GPT Images API 网络错误，准备重试: {exc}")
            except Exception as exc:
                last_message = _truncate_message(str(exc))
                logger.exception("GPT Images API 调用异常")
                break

    return None, None, last_message or "GPT Images API 调用失败"
