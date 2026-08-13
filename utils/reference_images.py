import base64
import binascii
import io
import re
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import unquote, urlparse

from PIL import Image as PillowImage


_DATA_URL_RE = re.compile(
    r"^data:([^;,]+);base64,(.*)$",
    re.IGNORECASE | re.DOTALL,
)


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
    if raw.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        brand = raw[8:12].lower()
        if brand in (b"heic", b"heix", b"hevc", b"hevx"):
            return "image/heic"
        if brand in (b"heif", b"mif1", b"msf1"):
            return "image/heif"
    return None


def _decode_base64_value(value: Any) -> Tuple[bytes, Optional[str]]:
    if isinstance(value, bytes):
        detected = _detect_image_mime(value)
        if detected:
            return value, detected
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("参考图不是有效的图片或 Base64 数据") from exc

    text = str(value or "").strip()
    # 兼容部分旧适配器产生的 base64:data:image/... 形式。
    if text.lower().startswith("base64:data:"):
        text = text[len("base64:") :]

    declared_mime: Optional[str] = None
    data_match = _DATA_URL_RE.match(text)
    if data_match:
        declared_mime = data_match.group(1).lower()
        text = data_match.group(2)
    elif text.lower().startswith("base64://"):
        text = text[len("base64://") :]
    elif text.lower().startswith("base64:"):
        text = text[len("base64:") :]

    compact = re.sub(r"\s+", "", text)
    try:
        raw = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"参考图 Base64 无效: {exc}") from exc

    return raw, _detect_image_mime(raw) or declared_mime


def _convert_to_supported_image(raw: bytes, mime_type: Optional[str]) -> Tuple[bytes, str]:
    # 三种接口都稳定支持这些静态图片类型。GIF 即便被伪装成 PNG，
    # 也必须依据真实文件头转码，否则 Gemini 会以 image/gif 拒绝请求。
    if mime_type in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        return raw, "image/jpeg" if mime_type == "image/jpg" else mime_type

    try:
        with PillowImage.open(io.BytesIO(raw)) as image:
            image.seek(0)  # 动图只使用首帧作为参考图
            has_alpha = image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            )
            converted = image.convert("RGBA" if has_alpha else "RGB")
            output = io.BytesIO()
            converted.save(output, format="PNG")
            return output.getvalue(), "image/png"
    except Exception as exc:
        raise ValueError(f"不支持或无法解码的参考图片格式 ({mime_type or 'unknown'}): {exc}") from exc


def normalize_reference_image(value: Any) -> str:
    """将任意 Base64 图片规范化为 MIME 与内容一致的 Data URI。"""
    raw, mime_type = _decode_base64_value(value)
    raw, mime_type = _convert_to_supported_image(raw, mime_type)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _file_uri_to_path(value: str) -> str:
    parsed = urlparse(value)
    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    if re.match(r"^/[A-Za-z]:/", path):
        path = path[1:]
    return path


async def image_component_to_data_url(component: Any) -> str:
    """读取 AstrBot 图片组件，并输出可供多模态接口使用的 Data URI。"""
    errors = []
    converter = getattr(component, "convert_to_base64", None)
    if callable(converter):
        try:
            return normalize_reference_image(await converter())
        except Exception as exc:
            errors.append(f"convert_to_base64: {exc}")

    path_converter = getattr(component, "convert_to_file_path", None)
    if callable(path_converter):
        try:
            path = str(await path_converter())
            raw = Path(_file_uri_to_path(path) if path.startswith("file:") else path).read_bytes()
            return normalize_reference_image(raw)
        except Exception as exc:
            errors.append(f"convert_to_file_path: {exc}")

    for attr in ("path", "file", "url"):
        candidate = str(getattr(component, attr, None) or "").strip()
        if not candidate:
            continue
        try:
            if candidate.startswith(("data:", "base64:")):
                return normalize_reference_image(candidate)
            local_path = _file_uri_to_path(candidate) if candidate.startswith("file:") else candidate
            if Path(local_path).is_file():
                return normalize_reference_image(Path(local_path).read_bytes())
        except Exception as exc:
            errors.append(f"{attr}: {exc}")

    detail = "; ".join(errors) if errors else "图片组件中没有可读取的 file/url/path"
    raise ValueError(detail)
