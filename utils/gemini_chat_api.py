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
_DATA_URL_RE = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\((https?://[^)\s]+)\)", re.IGNORECASE)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_MESSAGE_CHARS else text[:_MAX_MESSAGE_CHARS] + "\n...(message truncated)"


def _build_chat_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1") or base.endswith("/v1beta/openai"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _normalize_data_url(value: str, default_mime: str = "image/png") -> str:
    raw = str(value or "").strip()
    if raw.startswith("data:image/"):
        return raw
    return f"data:{default_mime};base64,{raw}"


async def _save_image(raw: bytes, mime_type: str = "image/png") -> str:
    suffix = "jpg" if "jpeg" in mime_type.lower() or "jpg" in mime_type.lower() else "webp" if "webp" in mime_type.lower() else "png"
    images_dir = Path(__file__).parent.parent / "images"
    images_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = images_dir / f"gemini_chat_image_{stamp}_{uuid.uuid4().hex[:8]}.{suffix}"
    path.write_bytes(raw)
    return str(path)


async def _materialize_image(value: str, timeout_seconds: int) -> Tuple[Optional[str], Optional[str]]:
    candidate = str(value or "").strip()
    match = _DATA_URL_RE.search(candidate)
    if match:
        try:
            raw = base64.b64decode(re.sub(r"\s+", "", match.group(2)), validate=True)
            return None, await _save_image(raw, match.group(1))
        except (binascii.Error, ValueError):
            return None, None
    if candidate.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.get(candidate)
                response.raise_for_status()
            mime = response.headers.get("content-type", "image/png").split(";", 1)[0]
            return candidate, await _save_image(response.content, mime)
        except Exception as exc:
            logger.warning(f"下载 geminichat 返回图片失败: {exc}")
            return candidate, None
    return None, None


def _image_values_from_block(block: Any) -> List[str]:
    if not isinstance(block, dict):
        return []
    values: List[str] = []
    image_url = block.get("image_url")
    if isinstance(image_url, str):
        values.append(image_url)
    elif isinstance(image_url, dict) and image_url.get("url"):
        values.append(str(image_url["url"]))

    inline = block.get("inlineData") or block.get("inline_data")
    if isinstance(inline, dict) and inline.get("data"):
        mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
        values.append(_normalize_data_url(str(inline["data"]), str(mime)))

    block_type = str(block.get("type") or "").lower()
    if block.get("b64_json"):
        values.append(_normalize_data_url(str(block["b64_json"])))
    if block_type in ("image", "output_image") and block.get("data"):
        mime = block.get("mime_type") or block.get("mimeType") or "image/png"
        values.append(_normalize_data_url(str(block["data"]), str(mime)))
    if block.get("url") and (block_type in ("", "image", "image_url", "output_image")):
        values.append(str(block["url"]))
    return values


async def _parse_chat_response(
    data: Dict[str, Any], timeout_seconds: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            error = error.get("message") or error
        return None, None, _truncate(str(error or "Chat API 响应未包含 choices"))

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    image_values: List[str] = []
    text_parts: List[str] = []
    content = message.get("content")

    if isinstance(content, str):
        image_values.extend(match.group(0) for match in _DATA_URL_RE.finditer(content))
        image_values.extend(match.group(1) for match in _MARKDOWN_IMAGE_RE.finditer(content))
        cleaned = _DATA_URL_RE.sub("", _MARKDOWN_IMAGE_RE.sub("", content)).strip()
        compact = re.sub(r"\s+", "", cleaned)
        if len(compact) >= 512 and _BASE64_RE.fullmatch(cleaned):
            try:
                raw = base64.b64decode(compact, validate=True)
                if raw.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")) or (
                    raw.startswith(b"RIFF") and len(raw) >= 12 and raw[8:12] == b"WEBP"
                ):
                    image_values.append(_normalize_data_url(compact))
                    cleaned = ""
            except (binascii.Error, ValueError):
                pass
        if cleaned:
            text_parts.append(cleaned)
    elif isinstance(content, list):
        for block in content:
            image_values.extend(_image_values_from_block(block))
            if isinstance(block, dict) and block.get("text"):
                text_parts.append(str(block["text"]))

    images = message.get("images")
    if isinstance(images, list):
        for block in images:
            if isinstance(block, str):
                image_values.append(block)
            else:
                image_values.extend(_image_values_from_block(block))

    # Some gateways attach images directly to the choice or response object.
    for container in (choices[0], data):
        extra_images = container.get("images") if isinstance(container, dict) else None
        if isinstance(extra_images, list):
            for block in extra_images:
                if isinstance(block, str):
                    image_values.append(block)
                else:
                    image_values.extend(_image_values_from_block(block))

    for value in image_values:
        image_url, image_path = await _materialize_image(value, timeout_seconds)
        if image_path:
            return image_url, image_path, _truncate("\n".join(text_parts)) if text_parts else None
    message_text = "\n".join(text_parts).strip()
    return None, None, _truncate(message_text or "Chat API 返回了消息，但未找到可解析的图片")


def _chat_parameters(config: Optional[Dict[str, Any]], temperature: Optional[float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    aliases = {"topP": "top_p", "top_p": "top_p", "seed": "seed", "maxOutputTokens": "max_tokens", "max_tokens": "max_tokens"}
    if isinstance(config, dict):
        for key, value in config.items():
            mapped = aliases.get(key)
            if mapped and value is not None:
                result[mapped] = value
    if temperature is not None:
        result["temperature"] = temperature
    return result


async def generate_or_edit_image_gemini_chat(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 300,
    temperature: Optional[float] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Call a Gemini-capable OpenAI-compatible /chat/completions endpoint."""
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    keys = [str(key).strip() for key in api_keys if str(key).strip()]
    if not keys:
        return None, None, "未提供 geminichat API Key/密码"

    content: Any = prompt
    if input_images_b64:
        content = [{"type": "text", "text": prompt}]
        for image in input_images_b64:
            content.append({"type": "image_url", "image_url": {"url": _normalize_data_url(image)}})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text", "image"],
    }
    payload.update(_chat_parameters(generation_config, temperature))
    url = _build_chat_url(api_base)
    last_message: Optional[str] = None

    for api_key in keys:
        for attempt in range(max(1, max_retry_attempts)):
            if attempt:
                await asyncio.sleep(min(2 ** attempt, 10))
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    response = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    last_message = _truncate(response.text)
                    logger.warning(f"geminichat API 暂时失败 {response.status_code}，准备重试")
                    continue
                if response.status_code == 400 and "modalit" in response.text.lower() and "modalities" in payload:
                    # Some compatible gateways generate images by model name but reject this extension.
                    fallback_payload = dict(payload)
                    fallback_payload.pop("modalities", None)
                    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                        response = await client.post(
                            url,
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json=fallback_payload,
                        )
                if response.status_code < 200 or response.status_code >= 300:
                    try:
                        error = response.json().get("error", response.json())
                        if isinstance(error, dict):
                            error = error.get("message") or error
                        last_message = _truncate(str(error))
                    except Exception:
                        last_message = _truncate(response.text)
                    return None, None, last_message
                return await _parse_chat_response(response.json(), timeout_seconds)
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_message = _truncate(str(exc))
                logger.warning(f"geminichat 网络错误，准备重试: {exc}")
            except Exception as exc:
                logger.exception("geminichat API 调用异常")
                return None, None, _truncate(str(exc))
    return None, None, last_message or "geminichat API 调用失败"
