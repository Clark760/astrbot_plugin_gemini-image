import asyncio
import base64
import binascii
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import httpx
from astrbot.api import logger


class _State:
    def __init__(self):
        self.api_key_index = 0
        self._lock = asyncio.Lock()

    async def get_next_key(self, keys: List[str]) -> str:
        async with self._lock:
            if not keys:
                raise ValueError("API key list is empty")
            key = keys[self.api_key_index % len(keys)]
            return key

    async def rotate(self, keys: List[str]):
        async with self._lock:
            if keys:
                self.api_key_index = (self.api_key_index + 1) % len(keys)


_state = _State()
_MAX_MESSAGE_CHARS = 1200
_BASE64_CHARS_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_DATA_URL_RE = re.compile(r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)")
_MARKDOWN_DATA_URL_RE = re.compile(
    r"!\[[^\]]*]\(\s*(data:image/[a-zA-Z0-9.+-]+;base64,[^)]+)\s*\)",
    re.IGNORECASE,
)


async def _save_bytes(content: bytes, suffix: str = "png") -> str:
    plugin_root = Path(__file__).parent.parent
    images_dir = plugin_root / "images"
    images_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    file_path = images_dir / f"gemini_image_{ts}_{uid}.{suffix}"
    file_path.write_bytes(content)
    return str(file_path)


async def _decode_and_save_base64(data_b64: str, mime: Optional[str]) -> str:
    # strip data URL if present
    if data_b64.startswith("data:"):
        try:
            header, b64 = data_b64.split(",", 1)
            data_b64 = b64
        except Exception:
            pass
    normalized_b64 = re.sub(r"\s+", "", data_b64 or "")
    try:
        raw = base64.b64decode(normalized_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 image payload: {e}") from e
    suffix = "png"
    if mime:
        mime = mime.lower()
        if "jpeg" in mime:
            suffix = "jpg"
        elif "jpg" in mime:
            suffix = "jpg"
        elif "webp" in mime:
            suffix = "webp"
        elif "png" in mime:
            suffix = "png"
    return await _save_bytes(raw, suffix)


def _truncate_message(text: str, max_chars: int = _MAX_MESSAGE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(message truncated)"


def _detect_image_mime_from_bytes(raw: bytes) -> Optional[str]:
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


def _extract_data_url_payload(data_url: str) -> Tuple[Optional[str], Optional[str]]:
    m = _DATA_URL_RE.search(data_url or "")
    if not m:
        return None, None
    mime = m.group(1)
    b64_payload = m.group(2)
    return mime, b64_payload


async def _extract_image_from_text(text: str) -> Tuple[Optional[str], str]:
    if not text:
        return None, text

    md_match = _MARKDOWN_DATA_URL_RE.search(text)
    if md_match:
        mime, b64_payload = _extract_data_url_payload(md_match.group(1))
        if mime and b64_payload:
            image_path = await _decode_and_save_base64(b64_payload, mime)
            cleaned = (text[: md_match.start()] + text[md_match.end() :]).strip()
            return image_path, cleaned

    data_match = _DATA_URL_RE.search(text)
    if data_match:
        mime = data_match.group(1)
        b64_payload = data_match.group(2)
        image_path = await _decode_and_save_base64(b64_payload, mime)
        cleaned = (text[: data_match.start()] + text[data_match.end() :]).strip()
        return image_path, cleaned

    candidate = text.strip()
    compact = re.sub(r"\s+", "", candidate)
    if len(compact) >= 512 and _BASE64_CHARS_RE.fullmatch(candidate or ""):
        try:
            raw = base64.b64decode(compact, validate=True)
            mime = _detect_image_mime_from_bytes(raw)
            if mime:
                image_path = await _decode_and_save_base64(compact, mime)
                return image_path, ""
        except (binascii.Error, ValueError):
            pass

    return None, text


def _build_url(api_base: str, path: str, api_key: str, model: str, append_key_query: bool, extra_query: Optional[Dict[str, str]] = None) -> str:
    base = api_base.rstrip("/")
    # 支持 {model} 占位符
    if "{model}" in path:
        path = path.replace("{model}", model)
    path = path if path.startswith("/") else f"/{path}"
    # 追加查询参数
    query_items = []
    if append_key_query and api_key:
        query_items.append(("key", api_key))
    if extra_query:
        for k, v in extra_query.items():
            if v is not None:
                query_items.append((k, str(v)))
    if query_items:
        sep = "&" if "?" in base + path else "?"
        qs = "&".join([f"{k}={v}" for k, v in query_items])
        return f"{base}{path}{sep}{qs}"
    else:
        return f"{base}{path}"


async def generate_or_edit_image_gemini(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    endpoint_path: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 60,
    temperature: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    使用 gcli2api 的 generateContent 接口生图/改图（通过 parts 注入图片）。

    返回 (image_url, image_path, message_text)。image_url 可能为 None（当只返回内联 base64 时）。
    """
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    if not api_keys:
        logger.error("未提供 API 密码/口令")
        return None, None, "未提供 API 密码/口令"

    # 允许传入参考图片进行编辑或条件生成
    input_images_b64 = input_images_b64 or []

    last_message: Optional[str] = None

    for key_attempt in range(len(api_keys)):
        current_key = await _state.get_next_key(api_keys)

        for attempt in range(max_retry_attempts):
            if attempt > 0:
                # 指数退避
                await asyncio.sleep(min(2 ** attempt, 10))

            # 允许通过 ?key= 传参，并附带头部，适配 gcli2api 的灵活鉴权
            use_query_key = True if current_key else False
            url = _build_url(api_base, endpoint_path, current_key, model, use_query_key, None)
            headers = {"Content-Type": "application/json"}
            if current_key:
                headers["x-goog-api-key"] = current_key
                headers["Authorization"] = f"Bearer {current_key}"

            # 构造 generateContent 风格负载
            parts = []
            parts.append({"text": prompt})
            for b64 in input_images_b64:
                mime_type = "image/png"
                if b64.startswith("data:"):
                    try:
                        header, b64data = b64.split(",", 1)
                        if header.startswith("data:") and ";base64" in header:
                            mime_type = header[5: header.find(";")]
                        b64 = b64data
                    except Exception:
                        pass
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": b64
                    }
                })

            payload: Dict = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ]
            }
            # 附加温度（仅传入 temperature，不包含 topP 等）
            if temperature is not None:
                gen = payload.get("generationConfig", {})
                gen2 = payload.get("generation_config", {})
                gen["temperature"] = temperature
                gen2["temperature"] = temperature
                payload["generationConfig"] = gen
                payload["generation_config"] = gen2

            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 429:
                        logger.warning("Gemini API 限流，稍后重试")
                        continue

                    if resp.status_code >= 500:
                        logger.warning(f"Gemini API 服务端错误 {resp.status_code}")
                        continue

                    if resp.status_code != 200:
                        # 4xx 明确错误不再重试当前密钥
                        try:
                            err = resp.json()
                            last_message = _truncate_message(str(err.get("error", err)))
                        except Exception:
                            last_message = _truncate_message(resp.text)
                            err = {"text": resp.text}
                        logger.error(f"Gemini API 调用失败 {resp.status_code}: {err}")
                        return None, None, last_message

                    data = resp.json()

                    # 解析 generateContent 返回结构
                    if isinstance(data, dict):
                        if data.get("error"):
                            last_message = _truncate_message(str(data.get("error")))
                            logger.error(f"Gemini API 返回错误: {data['error']}")
                            return None, None, last_message

                    image_url, image_path, message_text = await _parse_generate_content_json_for_image(data)
                    last_message = message_text or last_message

                    if image_path:
                        return image_url, image_path, message_text

                    if message_text:
                        logger.info("Gemini API 返回文本信息，无可用图片")
                        return image_url, None, message_text

                    # 没有图片也没有文本，直接把原始 JSON 返回给用户
                    try:
                        raw_text = json.dumps(data, ensure_ascii=False)
                        last_message = _truncate_message(raw_text)
                    except Exception:
                        last_message = last_message or "Gemini API 响应未包含可解析的图片或文本数据"
                    logger.error("Gemini API 响应未包含可解析的图片或文本数据")
                    return None, None, last_message

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_message = _truncate_message(str(e))
                logger.error(f"网络错误: {e}")
                continue
            except Exception as e:
                last_message = _truncate_message(str(e))
                logger.error(f"调用 Gemini API 异常: {e}")
                continue

        # 尝试下一个密钥
        await _state.rotate(api_keys)

    logger.error("所有 API 密钥与重试次数用尽，生成失败")
    if not last_message:
        last_message = "所有 API 密钥与重试次数用尽，生成失败"
    return None, None, last_message


async def _parse_generate_content_json_for_image(data: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """从 generateContent 风格响应解析 inlineData 图片与文本，返回 (url, path, message_text)。"""
    image_path = None
    image_url = None
    texts = []
    try:
        cands = data.get("candidates") or []
        for cand in cands:
            parts = (cand.get("content") or {}).get("parts") or []
            for p in parts:
                inline = p.get("inline_data") or p.get("inlineData")
                if inline and inline.get("data") and not image_path:
                    image_path = await _decode_and_save_base64(
                        inline.get("data"), inline.get("mime_type") or inline.get("mimeType")
                    )
                text_part = p.get("text")
                if text_part:
                    parsed_path, cleaned_text = await _extract_image_from_text(str(text_part))
                    if parsed_path and not image_path:
                        image_path = parsed_path
                    if cleaned_text and cleaned_text.strip():
                        texts.append(cleaned_text.strip())
        if not texts and data.get("error"):
            err = data.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or str(err)
                texts.append(msg)
            else:
                texts.append(str(err))
    except Exception as e:
        logger.warning(f"解析 generateContent 响应失败: {e}")
    message_text = _truncate_message("\n".join(texts)) if texts else None
    return image_url, image_path, message_text


async def generate_or_edit_image_gemini_stream(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    endpoint_path: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 60,
    temperature: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    调用流式接口（streamGenerateContent）。收到第一帧图片即返回。
    失败时返回 (None, None, message_text)。
    """
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    if not api_keys:
        logger.error("未提供 API 密钥/口令")
        return None, None, "未提供 API 密钥/口令"

    input_images_b64 = input_images_b64 or []
    last_message: Optional[str] = None

    for key_attempt in range(len(api_keys)):
        current_key = await _state.get_next_key(api_keys)

        for attempt in range(max_retry_attempts):
            if attempt > 0:
                await asyncio.sleep(min(2 ** attempt, 10))

            use_query_key = True if current_key else False
            url = _build_url(api_base, endpoint_path, current_key, model, use_query_key, {"alt": "sse"})
            headers = {"Content-Type": "application/json"}
            if current_key:
                headers["x-goog-api-key"] = current_key
                headers["Authorization"] = f"Bearer {current_key}"

            # 构造 generateContent 风格负载
            parts = [{"text": prompt}]
            for b64 in input_images_b64:
                mime_type = "image/png"
                if b64.startswith("data:"):
                    try:
                        header, b64data = b64.split(",", 1)
                        if header.startswith("data:") and ";base64" in header:
                            mime_type = header[5: header.find(";")]
                        b64 = b64data
                    except Exception:
                        pass
                parts.append({"inlineData": {"mimeType": mime_type, "data": b64}})

            payload: Dict = {
                "contents": [{"role": "user", "parts": parts}]
            }
            if temperature is not None:
                gen = payload.get("generationConfig", {})
                gen2 = payload.get("generation_config", {})
                gen["temperature"] = temperature
                gen2["temperature"] = temperature
                payload["generationConfig"] = gen
                payload["generation_config"] = gen2

            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as resp:
                        # 非 200 直接解析一次错误文本
                        if resp.status_code != 200:
                            try:
                                err_text = await resp.aread()
                                last_message = _truncate_message(err_text.decode("utf-8", errors="ignore"))
                                logger.error(f"流式接口状态 {resp.status_code}: {err_text[:200]}")
                            except Exception:
                                logger.error(f"流式接口状态 {resp.status_code}")
                            return None, None, last_message

                        ctype = resp.headers.get("content-type", "")
                        # SSE: text/event-stream; charset=utf-8
                        if "text/event-stream" in ctype:
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith(":"):
                                    # SSE 注释行
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str in ("[DONE]", "DONE"):
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                    except Exception:
                                        if not last_message:
                                            last_message = _truncate_message(data_str)
                                        continue
                                    # 错误帧
                                    if isinstance(data_json, dict) and data_json.get("error"):
                                        last_message = _truncate_message(str(data_json.get("error")))
                                        logger.error(f"流式错误帧: {data_json.get('error')}")
                                        break
                                    image_url, image_path, message_text = await _parse_generate_content_json_for_image(data_json)
                                    if message_text:
                                        last_message = message_text
                                    if image_path:
                                        return image_url, image_path, message_text
                        else:
                            # 非 SSE：尝试按 chunk/换行分割 JSON
                            buf = b""
                            async for chunk in resp.aiter_bytes():
                                buf += chunk
                                # 尝试按换行切分
                                while b"\n" in buf:
                                    line, buf = buf.split(b"\n", 1)
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        data_json = json.loads(line.decode("utf-8", errors="ignore"))
                                    except Exception:
                                        if not last_message:
                                            last_message = _truncate_message(line.decode("utf-8", errors="ignore"))
                                        continue
                                    if isinstance(data_json, dict) and data_json.get("error"):
                                        last_message = _truncate_message(str(data_json.get("error")))
                                        logger.error(f"流式分块错误: {data_json.get('error')}")
                                        break
                                    image_url, image_path, message_text = await _parse_generate_content_json_for_image(data_json)
                                    if message_text:
                                        last_message = message_text
                                    if image_path:
                                        return image_url, image_path, message_text
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_message = _truncate_message(str(e))
                logger.error(f"流式网络错误: {e}")
                continue
            except Exception as e:
                last_message = _truncate_message(str(e))
                logger.error(f"流式调用异常: {e}")
                continue

        await _state.rotate(api_keys)

    logger.error("流式接口未返回图片数据")
    if last_message:
        return None, None, last_message
    return None, None, "流式接口未返回图片或文本"
