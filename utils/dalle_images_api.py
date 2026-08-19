import asyncio
import base64
import binascii
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from astrbot.api import logger


_MAX_MESSAGE_CHARS = 1200
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
_DATA_URL_RE = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)",
    re.IGNORECASE,
)
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*]\(\s*(https?://[^)\s]+)\s*\)",
    re.IGNORECASE,
)
_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _truncate(text: str) -> str:
    value = str(text or "").strip()
    if len(value) <= _MAX_MESSAGE_CHARS:
        return value
    return value[:_MAX_MESSAGE_CHARS] + "\n...(message truncated)"


def _build_generations_url(api_base: str) -> str:
    base = str(api_base or "").rstrip("/")
    if base.endswith("/images/generations"):
        return base
    if base.endswith("/v1"):
        return f"{base}/images/generations"
    return f"{base}/v1/images/generations"


def _detect_image_mime(raw: bytes) -> Optional[str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"RIFF") and len(raw) >= 12 and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return None


def _suffix_for_mime(mime_type: Optional[str], url: str = "") -> str:
    mime = str(mime_type or "").lower()
    if "jpeg" in mime or "jpg" in mime:
        return "jpg"
    if "webp" in mime:
        return "webp"
    if "gif" in mime:
        return "gif"
    if "bmp" in mime:
        return "bmp"
    path = urlparse(url).path.lower() if url else ""
    for suffix in ("png", "jpg", "jpeg", "webp", "gif", "bmp"):
        if path.endswith(f".{suffix}"):
            return "jpg" if suffix == "jpeg" else suffix
    return "png"


async def _save_image(raw: bytes, mime_type: Optional[str], source_url: str = "") -> str:
    images_dir = Path(__file__).parent.parent / "images"
    images_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _suffix_for_mime(mime_type, source_url)
    path = images_dir / f"dalle_image_{stamp}_{uuid.uuid4().hex[:8]}.{suffix}"
    path.write_bytes(raw)
    return str(path)


def _is_explicit_image_url(value: str) -> bool:
    try:
        return urlparse(value).path.lower().endswith(_IMAGE_EXTENSIONS)
    except Exception:
        return False


def _extract_image_candidates(data: Any) -> List[str]:
    """Extract image data URLs/base64 values and HTTP URLs from irregular responses."""
    candidates: List[str] = []
    seen = set()

    def add(value: Any):
        candidate = str(value or "").strip().rstrip(".,;]}>")
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    def scan_text(value: str):
        text = str(value or "")
        for match in _DATA_URL_RE.finditer(text):
            add(match.group(0))
        for match in _MARKDOWN_IMAGE_RE.finditer(text):
            add(match.group(1))
        for match in _HTTP_URL_RE.finditer(text):
            url = match.group(0).rstrip(").,;]}>")
            if _is_explicit_image_url(url):
                add(url)

    def walk(value: Any, parent_key: str = ""):
        if isinstance(value, dict):
            # Prefer standard structured image fields before scanning arbitrary text.
            for key in ("b64_json", "url", "image_url"):
                item = value.get(key)
                if isinstance(item, str):
                    if key == "b64_json":
                        add(f"base64:{item}")
                    elif item.startswith(("http://", "https://", "data:image/")):
                        add(item)
                elif isinstance(item, dict) and isinstance(item.get("url"), str):
                    add(item["url"])
            for key, item in value.items():
                walk(item, str(key).lower())
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, parent_key)
        elif isinstance(value, str):
            if parent_key in ("b64_json", "base64", "image_base64"):
                add(f"base64:{value}")
            elif parent_key in ("url", "image_url") and value.startswith(("http://", "https://", "data:image/")):
                add(value)
            scan_text(value)

    walk(data)
    return candidates


async def _materialize_candidate(
    candidate: str,
    timeout_seconds: int,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[Optional[str], Optional[str]]:
    data_match = _DATA_URL_RE.fullmatch(candidate.strip())
    if data_match:
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", data_match.group(2)), validate=True)
        except (binascii.Error, ValueError):
            return None, None
        detected = _detect_image_mime(raw)
        if not detected:
            return None, None
        return None, await _save_image(raw, detected)

    if candidate.startswith("base64:"):
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", candidate[7:]), validate=True)
        except (binascii.Error, ValueError):
            return None, None
        detected = _detect_image_mime(raw)
        if not detected:
            return None, None
        return None, await _save_image(raw, detected)

    if not candidate.startswith(("http://", "https://")):
        return None, None

    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True)
    try:
        response = await active_client.get(candidate)
        response.raise_for_status()
        if len(response.content) > _MAX_DOWNLOAD_BYTES:
            logger.warning(f"DALL-E 返回图片超过 {_MAX_DOWNLOAD_BYTES // 1024 // 1024}MB，已跳过: {candidate}")
            return candidate, None
        header_mime = response.headers.get("content-type", "").split(";", 1)[0].lower()
        detected = _detect_image_mime(response.content)
        if not detected and not header_mime.startswith("image/"):
            logger.warning(f"DALL-E 返回链接不是图片，已跳过: {candidate}")
            return candidate, None
        mime_type = detected or header_mime
        return candidate, await _save_image(response.content, mime_type, candidate)
    except Exception as exc:
        logger.warning(f"下载 DALL-E 返回图片失败: {candidate} | {exc}")
        return candidate, None
    finally:
        if owns_client:
            await active_client.aclose()


async def _parse_dalle_response(
    data: Any,
    raw_text: str,
    timeout_seconds: int,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    candidates = _extract_image_candidates(data)
    if raw_text:
        for candidate in _extract_image_candidates(raw_text):
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        image_url, image_path = await _materialize_candidate(candidate, timeout_seconds, client)
        if image_path:
            # Finding an image is success; suppress markdown/text wrappers from the gateway.
            return image_url, image_path, None

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            error = error.get("message") or error
        if error:
            return None, None, _truncate(str(error))

    return None, None, _truncate(raw_text or str(data) or "DALL-E API 响应未包含可解析的图片")


def _filter_dalle_parameters(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    aliases = {
        "size": "size",
        "aspect_ratio": "aspect_ratio",
        "aspectRatio": "aspect_ratio",
        "image_size": "image_size",
        "imageSize": "image_size",
        "quality": "quality",
        "n": "n",
        "seed": "seed",
        "response_format": "response_format",
        "responseFormat": "response_format",
        "output_format": "output_format",
        "outputFormat": "output_format",
    }
    result: Dict[str, Any] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            mapped = aliases.get(key)
            if mapped and value is not None:
                result[mapped] = value
    return result


async def generate_or_edit_image_dalle(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 300,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Call a gateway's extended DALL-E Generations endpoint for text/image-to-image."""
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    keys = [str(key).strip() for key in api_keys if str(key).strip()]
    if not keys:
        return None, None, "未提供 DALL-E API Key/密码"

    payload: Dict[str, Any] = {"model": model, "prompt": prompt}
    payload.update(_filter_dalle_parameters(generation_config))
    reference_images = input_images_b64 or []
    if reference_images:
        payload["image"] = reference_images

    url = _build_generations_url(api_base)
    last_message: Optional[str] = None
    logger.info(
        f"DALL-E Generations 请求：参考图 {len(reference_images)} 张，"
        f"参数={list(_filter_dalle_parameters(generation_config).keys())}"
    )

    for api_key in keys:
        for attempt in range(max(1, max_retry_attempts)):
            if attempt:
                await asyncio.sleep(min(2 ** attempt, 10))
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
                    response = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                    if response.status_code == 429 or response.status_code >= 500:
                        last_message = _truncate(response.text)
                        logger.warning(f"DALL-E API 暂时失败 {response.status_code}，准备重试")
                        continue
                    if response.status_code < 200 or response.status_code >= 300:
                        try:
                            error_data = response.json()
                            error = error_data.get("error", error_data)
                            if isinstance(error, dict):
                                error = error.get("message") or error
                            last_message = _truncate(str(error))
                        except Exception:
                            last_message = _truncate(response.text)
                        return None, None, last_message

                    try:
                        data: Any = response.json()
                    except Exception:
                        data = response.text
                    return await _parse_dalle_response(data, response.text, timeout_seconds, client)
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_message = _truncate(str(exc))
                logger.warning(f"DALL-E API 网络错误，准备重试: {exc}")
            except Exception as exc:
                logger.exception("DALL-E API 调用异常")
                return None, None, _truncate(str(exc))

    return None, None, last_message or "DALL-E API 调用失败"
