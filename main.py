from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply, Plain, Image
from typing import Optional, Dict, Tuple, Any, List
import time
import os
from pathlib import Path
import shutil
import importlib
import shlex
import json

try:
    _command_module = importlib.import_module("astrbot.core.star.filter.command")
    GreedyStr = getattr(_command_module, "GreedyStr")
except Exception:  # AstrBot 未安装时的开发环境降级
    class GreedyStr(str):
        pass

from .utils.file_send_server import send_file
from .utils.reference_images import image_component_to_data_url


@register("gemini-image", "薄暝", "支持 Gemini 与 GPT 的生图/改图并发送到 QQ", "0.7.4")
class GeminiImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        self.provider = self._normalize_provider(config.get("provider", "gemini"))
        # 保留原配置名以兼容旧版本；GPT 模式下它表示 OpenAI 兼容 API Base。
        default_base = (config.get("gcli2api_base_url") or "http://127.0.0.1:7861").strip()
        # 固定端点（强制 v1beta），不再提供配置项
        self.api_base = default_base
        self._GEN_PATH = "/v1beta/models/{model}:generateContent"
        self._STREAM_GEN_PATH = "/v1beta/models/{model}:streamGenerateContent"

        # 模型与重试
        self.model_name = (config.get("model_name") or "").strip()
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))
        # 固定策略：默认启用流式，附带 alt=sse；不提供开关
        self.use_stream = True
        # 温度参数（重新加入配置）
        try:
            self.temperature = float(config.get("temperature", 1.0))
        except Exception:
            self.temperature = 1.0

        # 请求超时时间（适配大图耗时场景）
        try:
            self.request_timeout_seconds = int(config.get("request_timeout_seconds", 300))
        except Exception:
            self.request_timeout_seconds = 300

        # gcli2api 鉴权（默认 pwd）
        self.gcli2api_api_password = (config.get("gcli2api_api_password") or "pwd").strip()

        # 群控制与限流
        self.group_control_mode = (config.get("group_control_mode") or "off").strip().lower()
        self.group_list = list(config.get("group_list", []))
        try:
            self.group_rate_window_seconds = int(config.get("group_rate_window_seconds", 3600))
        except Exception:
            self.group_rate_window_seconds = 3600
        try:
            self.group_rate_max_calls = int(config.get("group_rate_max_calls", 10))
        except Exception:
            self.group_rate_max_calls = 10
        # 运行时计数：group_id -> {"window_start": float, "count": int}
        self._group_call_bucket = {}

        # NapCat 自定义 TCP 文件中转（可选，默认关闭）。这不是 NapCat 自带端口；
        # Docker 共享 /AstrBot/data 时应直接使用共享路径，无需开启。
        self.nap_file_forward_enabled = bool(config.get("nap_file_forward_enabled", False))
        self.nap_server_address = str(config.get("nap_server_address") or "").strip()
        try:
            self.nap_server_port = int(config.get("nap_server_port") or 0)
        except (TypeError, ValueError):
            self.nap_server_port = 0

        self._global_config_loaded = False

    @staticmethod
    def _normalize_provider(provider: Any) -> str:
        value = str(provider or "gemini").strip().lower()
        return value if value in ("gemini", "geminichat", "gpt") else "gemini"

    def _is_private_chat(self, event: AstrMessageEvent) -> bool:
        """更稳健地判断是否私聊，避免仅依赖 group_id 导致误判。"""
        # 1) 优先使用框架可能提供的 is_private_chat
        try:
            checker = getattr(event, "is_private_chat", None)
            if callable(checker):
                val = checker()
                if isinstance(val, bool):
                    return val
        except Exception:
            pass

        # 2) 其次根据 message_obj.type 判断
        try:
            msg_obj = getattr(event, "message_obj", None)
            msg_type = getattr(msg_obj, "type", None)
            if msg_type is not None:
                msg_type_str = str(getattr(msg_type, "value", msg_type)).lower()
                if "private" in msg_type_str or "friend" in msg_type_str:
                    return True
                if "group" in msg_type_str:
                    return False
        except Exception:
            pass

        # 3) 最后回退 group_id 规则
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        gid_str = "" if gid is None else str(gid).strip().lower()
        return gid_str in ("", "0", "none", "null")

    async def _load_global_config(self):
        if self._global_config_loaded:
            return
        try:
            plugin_config = await sp.global_get("gemini-image", {})
            if "provider" in plugin_config:
                self.provider = self._normalize_provider(plugin_config.get("provider"))
                logger.info(f"从全局配置加载 provider: {self.provider}")
            if "gcli2api_base_url" in plugin_config:
                self.api_base = str(plugin_config["gcli2api_base_url"]).strip() or self.api_base
                logger.info(f"从全局配置加载 gcli2api_base_url: {self.api_base}")
            if "model_name" in plugin_config:
                self.model_name = str(plugin_config["model_name"]).strip() or self.model_name
                logger.info(f"从全局配置加载 model_name: {self.model_name}")
            # 不再加载端点与流式相关配置项（固定策略）
            if "gcli2api_api_password" in plugin_config:
                self.gcli2api_api_password = str(plugin_config["gcli2api_api_password"]).strip() or self.gcli2api_api_password
            # 群控制
            if "group_control_mode" in plugin_config:
                self.group_control_mode = str(plugin_config.get("group_control_mode", self.group_control_mode) or "").strip().lower()
            if "group_list" in plugin_config and isinstance(plugin_config.get("group_list"), list):
                self.group_list = list(plugin_config.get("group_list", self.group_list))
            if "group_rate_window_seconds" in plugin_config:
                try:
                    self.group_rate_window_seconds = int(plugin_config.get("group_rate_window_seconds", self.group_rate_window_seconds))
                except Exception:
                    pass
            if "group_rate_max_calls" in plugin_config:
                try:
                    self.group_rate_max_calls = int(plugin_config.get("group_rate_max_calls", self.group_rate_max_calls))
                except Exception:
                    pass
            # 重新加载温度配置（其余生成参数固定不提供）
            if "temperature" in plugin_config:
                try:
                    self.temperature = float(plugin_config.get("temperature", self.temperature))
                except Exception:
                    pass
            if "request_timeout_seconds" in plugin_config:
                try:
                    self.request_timeout_seconds = int(plugin_config.get("request_timeout_seconds", self.request_timeout_seconds))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
        finally:
            self._global_config_loaded = True

    async def _get_user_config(self, event: AstrMessageEvent) -> Optional[dict]:
        """获取用户个人配置（仅私聊）"""
        try:
            if not self._is_private_chat(event):
                return None
            
            # 私聊，尝试获取用户ID
            user_id = None
            try:
                user_id = event.get_sender_id()
            except Exception:
                return None
            
            if not user_id:
                return None
            
            # 获取用户配置
            user_config_key = f"gemini-image-user-{user_id}"
            user_config = await sp.global_get(user_config_key, {})
            
            return user_config if user_config else None
        except Exception as e:
            logger.warning(f"获取用户配置失败: {e}")
            return None

    async def _save_user_config(self, event: AstrMessageEvent, provider: str, api_base: str, api_password: str, model_name: str) -> bool:
        """保存用户个人配置（仅私聊）"""
        try:
            user_id = None
            try:
                user_id = event.get_sender_id()
            except Exception:
                return False
            
            if not user_id:
                return False
            
            user_config_key = f"gemini-image-user-{user_id}"
            # 保留已有配置，更新本次传入字段
            user_config = await sp.global_get(user_config_key, {})
            user_config["provider"] = self._normalize_provider(provider)
            user_config["api_base"] = api_base.strip()
            user_config["api_password"] = api_password.strip()
            user_config["model_name"] = model_name.strip()
            
            await sp.global_put(user_config_key, user_config)
            logger.info(f"已保存用户 {user_id} 的配置")
            return True
        except Exception as e:
            logger.error(f"保存用户配置失败: {e}")
            return False

    def _private_config_required_message(self) -> str:
        return (
            "❌ 私聊使用需要先配置个人API\n\n"
            "请使用以下命令设置：\n"
            "/设置ai配置 <gemini|geminichat|gpt> <api_base> <api_key> <model_name>\n\n"
            "例如：\n"
            "/设置ai配置 gemini http://127.0.0.1:7861 your_password gemini-3-pro-image\n"
            "/设置ai配置 geminichat http://127.0.0.1:7861 your_password gemini-3-pro-image\n"
            "/设置ai配置 gpt https://api.openai.com sk-xxx gpt-image-2\n\n"
            "查看当前配置请使用：/查看ai配置"
        )

    async def _ensure_private_has_config(self, event: AstrMessageEvent):
        """若为私聊则确保已有个人配置，否则返回提示组件"""
        if self._is_private_chat(event):
            user_cfg = await self._get_user_config(event)
            if not user_cfg:
                return False, event.plain_result(self._private_config_required_message())
        return True, None

    def _check_group_access(self, event: AstrMessageEvent) -> Optional[str]:
        """检查群白/黑名单与限流，返回错误提示或 None 允许通过"""
        try:
            gid = None
            try:
                gid = event.get_group_id()  # 群聊返回群号，私聊返回 None
            except Exception:
                gid = None

            # 白/黑名单
            mode = self.group_control_mode
            if gid:
                if mode == "whitelist" and gid not in self.group_list:
                    return "当前群未被授权使用本插件"
                if mode == "blacklist" and gid in self.group_list:
                    return "当前群已被限制使用本插件"

                # 限流：仅对群聊生效
                import time
                now = time.time()
                b = self._group_call_bucket.get(gid, {"window_start": now, "count": 0})
                window_start = b.get("window_start", now)
                count = int(b.get("count", 0))
                if now - window_start >= self.group_rate_window_seconds:
                    window_start = now
                    count = 0
                if count >= self.group_rate_max_calls:
                    return "本群调用已达上限，请稍后再试"
                # 预占位+1（通过后真正执行业务）
                b["window_start"], b["count"] = window_start, count + 1
                self._group_call_bucket[gid] = b
            else:
                # 私聊不做名单与限流限制
                pass
        except Exception:
            # 出错不拦截
            return None
        return None

    async def resolve_image_target_for_send(self, image_path: str) -> str:
        callback_api_base = self.context.get_config().get("callback_api_base")
        if not callback_api_base:
            return image_path
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.convert_to_web_link()
            return download_url
        except Exception as e:
            logger.warning(f"回退本地文件发送，原因: {e}")
            return image_path

    async def gemini_image_tool(
        self,
        event: AstrMessageEvent,
        image_description: str,
        use_reference_images: bool = True,
        mode: str = "auto",
        appended_generation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate or edit images via Gemini generateContent or OpenAI Images API.
        If images exist in the message/reply and use_reference_images=True, will include them.
        mode: "auto" | "generate" | "edit". When "auto", edit if references provided else generate.
        """
        await self._load_global_config()
        # 尝试执行 images 目录定期清理
        await self._maybe_cleanup_images()

        # 检查是否为私聊，如果是则尝试加载用户个人配置
        provider = self.provider
        api_base = self.api_base
        api_password = self.gcli2api_api_password
        configured_model = self.model_name
        
        user_config = await self._get_user_config(event)
        if user_config:
            # 使用用户个人配置
            provider = self._normalize_provider(user_config.get("provider", "gemini"))
            api_base = user_config.get("api_base", self.api_base)
            api_password = user_config.get("api_password", self.gcli2api_api_password)
            configured_model = user_config.get("model_name", "")
            logger.info(f"使用用户个人配置: {api_base} | 类型: {provider}")
        else:
            # 检查是否为私聊但没有配置
            if self._is_private_chat(event):
                # 私聊但没有个人配置
                yield event.plain_result(self._private_config_required_message())
                return

        model_name = str(configured_model or "").strip()
        if not model_name:
            scope = "个人配置" if user_config else "插件全局设置"
            yield event.plain_result(f"❌ {scope}缺少必填的 model_name，请先填写实际模型名称。")
            return

        # 为提示词添加图片生成目标提示，避免多模态模型返回纯文本
        image_generation_prefix = "【本次任务目标：生成图片】请根据以下描述生成一张图片，必须输出图像而非文本描述：\n"
        image_description = image_generation_prefix + image_description

        # api_password 在 Gemini 模式下可为代理密码，在 GPT 模式下通常为 API Key。

        # 收集并规范化参考图片。按真实文件头识别格式，GIF 等会自动转成 PNG。
        input_images: List[str] = []
        reference_components = self._get_reference_image_components(event) if use_reference_images else []
        for index, component in enumerate(reference_components, start=1):
            try:
                input_images.append(await image_component_to_data_url(component))
            except Exception as e:
                logger.warning(f"第 {index} 张参考图片读取/转换失败: {e}")

        if use_reference_images:
            logger.info(
                f"参考图处理完成：识别 {len(reference_components)} 张，成功读取 {len(input_images)} 张；"
                f"接口类型={provider}，模式={mode}"
            )
        if mode == "edit" and not input_images:
            yield event.plain_result(
                "❌ 参考图片读取或格式转换失败，已停止本次改图，未降级为普通生图。"
                "请尝试重新发送 JPG/PNG/WEBP 图片，并查看日志中的“参考图片读取/转换失败”详情。"
            )
            return

        merged_generation_config: Dict[str, Any] = dict(appended_generation_config or {})
        effective_temperature = self.temperature
        if "temperature" in merged_generation_config:
            try:
                effective_temperature = float(merged_generation_config.pop("temperature"))
            except Exception:
                yield event.plain_result("参数错误：`--temperature` 必须是数字。")
                return

        # Gemini 使用 generateContent；GPT 使用 OpenAI Images API。
        endpoint_path = self._STREAM_GEN_PATH if self.use_stream else self._GEN_PATH

        # 记录开始时间
        start_time = time.time()
        
        try:
            message_text = None
            if provider == "gpt":
                from .utils.openai_images_api import generate_or_edit_image_openai
                image_url, image_path, message_text = await generate_or_edit_image_openai(
                    prompt=image_description,
                    api_keys=[api_password] if api_password else [],
                    model=model_name,
                    api_base=api_base,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                    generation_config=merged_generation_config or None,
                    timeout_seconds=self.request_timeout_seconds,
                )
            elif provider == "geminichat":
                from .utils.gemini_chat_api import generate_or_edit_image_gemini_chat
                image_url, image_path, message_text = await generate_or_edit_image_gemini_chat(
                    prompt=image_description,
                    api_keys=[api_password] if api_password else [],
                    model=model_name,
                    api_base=api_base,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                    temperature=effective_temperature,
                    generation_config=merged_generation_config or None,
                    timeout_seconds=self.request_timeout_seconds,
                )
            elif self.use_stream:
                from .utils.gemini_images_api import generate_or_edit_image_gemini_stream
                image_url, image_path, message_text = await generate_or_edit_image_gemini_stream(
                    prompt=image_description,
                    api_keys=[api_password] if api_password else [""],
                    model=model_name,
                    api_base=api_base,
                    endpoint_path=endpoint_path,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                    temperature=effective_temperature,
                    generation_config=merged_generation_config or None,
                    timeout_seconds=self.request_timeout_seconds,
                )
                # 流式失败则回退非流式
                if not image_path:
                    from .utils.gemini_images_api import generate_or_edit_image_gemini
                    image_url, image_path, message_text = await generate_or_edit_image_gemini(
                        prompt=image_description,
                        api_keys=[api_password] if api_password else [""],
                        model=model_name,
                        api_base=api_base,
                        endpoint_path=self._GEN_PATH,
                        input_images_b64=input_images,
                        max_retry_attempts=self.max_retry_attempts,
                        temperature=effective_temperature,
                        generation_config=merged_generation_config or None,
                        timeout_seconds=self.request_timeout_seconds,
                    )
            else:
                from .utils.gemini_images_api import generate_or_edit_image_gemini
                image_url, image_path, message_text = await generate_or_edit_image_gemini(
                    prompt=image_description,
                    api_keys=[api_password] if api_password else [""],
                    model=model_name,
                    api_base=api_base,
                    endpoint_path=endpoint_path,
                    input_images_b64=input_images,
                    max_retry_attempts=self.max_retry_attempts,
                    temperature=effective_temperature,
                    generation_config=merged_generation_config or None,
                    timeout_seconds=self.request_timeout_seconds,
                )

            if not image_path:
                if message_text:
                    yield event.plain_result(message_text)
                else:
                    yield event.plain_result("图像生成失败，请检查 API 配置与模型名称。")
                return

            # 计算耗时
            elapsed = time.time() - start_time

            # 可选：通过自定义 TCP 文件服务器中转。仅在显式启用后连接。
            if (
                self.nap_file_forward_enabled
                and self.nap_server_address
                and self.nap_server_port > 0
            ):
                try:
                    new_path = await send_file(image_path, self.nap_server_address, self.nap_server_port)
                    if new_path:
                        image_path = new_path
                except Exception as e:
                    logger.warning(f"Napcat 文件中转失败，回退为本地发送: {e}")

            image_target = await self.resolve_image_target_for_send(image_path)
            
            # 构建成功消息
            success_msg = f"✅ 生成成功 ({elapsed:.2f}s)"
            
            # 避免在私聊中触发 Node/合并转发路径导致 NapCat 图片发送失败
            if hasattr(event, "image_result"):
                yield event.image_result(image_target)
            else:
                # 兼容旧版 AstrBot：无 image_result 时仍尽量发送图片组件
                if str(image_target).startswith(("http://", "https://")):
                    yield event.chain_result([Image.fromURL(image_target)])
                else:
                    yield event.chain_result([Image.fromFileSystem(image_target)])
            yield event.plain_result(success_msg)
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{provider} 生图/改图异常: {e}")
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {str(e)}")

    async def _maybe_cleanup_images(self):
        """按配置每隔 N 天清理一次 images 目录（清空目录）。"""
        try:
            cfg = self.context.get_config() or {}
            if not bool(cfg.get("images_cleanup_enabled", True)):
                return
            days = int(cfg.get("images_cleanup_interval_days", 3))
            days = max(1, days)
            interval = days * 86400
            meta = await sp.global_get("gemini-image", {})
            last_ts = float(meta.get("images_cleanup_last_ts", 0))
            now = time.time()
            if now - last_ts < interval:
                return
            # 执行清理
            await self._cleanup_images_dir()
            meta["images_cleanup_last_ts"] = now
            await sp.global_put("gemini-image", meta)
        except Exception as e:
            logger.warning(f"images 清理调度失败: {e}")

    async def _cleanup_images_dir(self):
        try:
            images_dir = Path(__file__).parent / "images"
            if not images_dir.exists() or not images_dir.is_dir():
                return
            removed = 0
            for p in images_dir.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                        removed += 1
                except Exception as ie:
                    logger.debug(f"删除 {p} 失败: {ie}")
            if removed > 0:
                logger.info(f"已清理 images 目录，共删除 {removed} 项")
        except Exception as e:
            logger.warning(f"清理 images 目录失败: {e}")

    @filter.command("生图")
    async def cmd_generate(self, event: AstrMessageEvent):
        """生图：/生图 <提示词>"""
        # 提取全文本输入，解决图文混排导致的 GreedyStr 失效
        prompt_text = self._get_full_text_input(event, "/生图")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return

        # 群控制与限流
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return

        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return

        # 先返回生成中提示
        display_prompt = prompt[:20] + '...' if len(prompt) > 20 else prompt
        yield event.plain_result(f"🎨 收到请求，正在生成 [{display_prompt}]...")

        # 然后执行生成并发送结果
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=False, mode="generate", appended_generation_config=appended_params):
            yield res

    @filter.command("改图")
    async def cmd_edit(self, event: AstrMessageEvent):
        """改图（需携带/引用图片）：/改图 <提示词>"""
        # 提取全文本输入
        prompt_text = self._get_full_text_input(event, "/改图")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return

        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        # 如果未携带/引用图片，提示用户
        has_image = self._check_has_image(event)
        if not has_image:
            yield event.plain_result("请先携带或引用一张图片后，再使用：/改图 <提示词>")
            return

        # 先返回生成中提示
        display_prompt = prompt[:20] + '...' if len(prompt) > 20 else prompt
        yield event.plain_result(f"🎨 收到请求，正在生成 [{display_prompt}]...")

        # 然后执行生成并发送结果
        async for res in self.gemini_image_tool(event, image_description=prompt, use_reference_images=True, mode="edit", appended_generation_config=appended_params):
            yield res

    @filter.command("手办化")
    async def cmd_figure(self, event: AstrMessageEvent):
        """手办化（需携带/引用图片）：/手办化 [描述]"""
        # 提取全文本输入
        prompt_text = self._get_full_text_input(event, "/手办化")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return

        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        default_prompt = (
            "将画面中的角色重塑为顶级收藏级树脂手办，全身动态姿势，置于角色主题底座；"
            "高精度材质，手工涂装，肌肤纹理与服装材质真实分明。"
            "戏剧性硬光为主光源，凸显立体感，无过曝；强效补光消除死黑，细节完整可见。"
            "背景为窗边景深模糊，侧后方隐约可见产品包装盒。"
            "博物馆级摄影质感，全身细节无损，面部结构精准。"
            "禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
        )
        final_prompt = f"{default_prompt}\n用户补充要求：{prompt}" if prompt else default_prompt

        # 检查是否包含图片
        has_image = self._check_has_image(event)
        if not has_image:
            yield event.plain_result("手办化需要携带或引用图片，请附图后再发送：/手办化")
            return

        # 先返回生成中提示
        yield event.plain_result("🎨 收到请求，正在生成 [手办化]...")
        
        # 然后执行生成并发送结果
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=True, mode="edit", appended_generation_config=appended_params):
            yield res

    @filter.command("coser化")
    async def cmd_coser(self, event: AstrMessageEvent):
        """动漫角色转真人 Coser（需携带/引用图片）：/coser化 [补充要求]"""
        prompt_text = self._get_full_text_input(event, "/coser化")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        if not self._check_has_image(event):
            yield event.plain_result("coser化需要携带或引用一张角色插画，请附图后再发送：/coser化")
            return
        default_prompt = (
            "以参考插画中的角色为唯一设计依据，生成真人 Coser 的高质量摄影照片。"
            "准确保留角色的发型、发色、服装结构、配饰、色彩和辨识特征，将二次元设计自然转化为真实可制作的妆造与服装材质。"
            "场景设在 Comiket 会场，采用自然的人像摄影、真实皮肤与布料质感、协调光影和浅景深。"
            "人物应为成年人，五官自然，肢体完整，不改变角色核心身份，不添加无关文字或水印。"
        )
        final_prompt = f"{default_prompt}\n用户补充要求：{prompt}" if prompt else default_prompt
        yield event.plain_result("🎨 收到请求，正在生成 [coser化]...")
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=True, mode="edit", appended_generation_config=appended_params):
            yield res

    @filter.command("生成角色设定")
    async def cmd_character_design(self, event: AstrMessageEvent):
        """生成角色设定图（需携带/引用图片）：/生成角色设定 [补充要求]"""
        prompt_text = self._get_full_text_input(event, "/生成角色设定")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        if not self._check_has_image(event):
            yield event.plain_result("生成角色设定需要携带或引用一张角色参考图，请附图后再发送：/生成角色设定")
            return
        default_prompt = (
            "根据参考图生成一张专业、完整、排版清晰的角色设定图（Character Design Sheet）。"
            "严格保持角色身份、面部、发型、服装、配饰和配色一致。"
            "画面应包含：身高与头身比例设定；正面、侧面、背面三视图；多种典型表情；常见动作姿势；服装与关键配饰细节拆解。"
            "使用干净的浅色背景和统一比例，分区明确，所有视图造型一致，适合动画、游戏或插画制作参考。"
        )
        final_prompt = f"{default_prompt}\n用户补充要求：{prompt}" if prompt else default_prompt
        yield event.plain_result("🎨 收到请求，正在生成 [角色设定]...")
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=True, mode="edit", appended_generation_config=appended_params):
            yield res

    @filter.command("文章信息图")
    async def cmd_article_infographic(self, event: AstrMessageEvent):
        """将当前或 QQ 引用消息中的文章转换为信息图。"""
        prompt_text = self._get_full_text_input(event, "/文章信息图")
        current_article, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return
        quoted_article = self._get_quoted_text_input(event)
        article_parts = []
        if quoted_article:
            article_parts.append(f"引用消息内容：\n{quoted_article}")
        if current_article:
            label = "命令附加内容" if quoted_article else "文章正文"
            article_parts.append(f"{label}：\n{current_article}")
        article = "\n\n".join(article_parts)
        if not article:
            yield event.plain_result(
                "请在命令后提供文章内容，或引用一条包含文章文字的 QQ 消息后发送：/文章信息图"
            )
            return
        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        final_prompt = (
            "请把下面的文章制作成一张结构清晰、易于快速阅读的信息图。"
            "先准确理解内容，将其翻译并提炼为简洁英文，保留核心观点、关键数据和逻辑关系。"
            "图中只使用必要的大标题、短标签和极简说明，避免大段文字；信息层级清晰，阅读顺序明确。"
            "加入丰富、可爱且与主题相关的卡通人物、图标和视觉元素，确保文字清晰可辨、事实忠于原文。\n"
            f"文章内容：\n{article}"
        )
        has_image = self._check_has_image(event)
        yield event.plain_result("🎨 收到请求，正在生成 [文章信息图]...")
        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate", appended_generation_config=appended_params):
            yield res

    @filter.command("提示词参考")
    async def cmd_prompt_reference(self, event: AstrMessageEvent):
        """返回 Nano Banana 提示词参考链接：/提示词参考"""
        yield event.plain_result("Nano Banana 提示词参考：\nhttps://github.com/newaiproxy/nanobanana-prompt")

    @filter.command("aiimg帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        """帮助：/aiimg帮助"""
        help_text = (
            "🎨 AI 图像生成插件完整帮助\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📝 基础指令：\n"
            "• /生图 <提示词>\n"
            "  纯文本生成图片，不读取消息中的参考图。\n"
            "• /改图 <提示词>\n"
            "  必须携带或引用图片，根据提示词编辑图片。\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🧑 角色指令（必须携带/引用图片）：\n"
            "• /手办化 [补充要求]\n"
            "  将角色转换为收藏级树脂手办。\n"
            "• /coser化 [补充要求]\n"
            "  将角色插画转换为真人 Coser 摄影图。\n"
            "• /生成角色设定 [补充要求]\n"
            "  生成比例、三视图、表情、动作、服装及配饰设定。\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🖼️ 风格指令（可携带/引用图片）：\n"
            "• /海报 [补充要求]\n"
            "  生成16:9电影宣传海报。\n"
            "• /表情包 [补充要求]\n"
            "  生成Q版 LINE 贴纸风格表情包。\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📊 内容与参考：\n"
            "• /文章信息图 [文章内容]\n"
            "  可直接输入文章，或引用一条 QQ 文字消息后发送本指令；\n"
            "  也可在指令后继续附加要求，并可附参考图。\n"
            "• /提示词参考\n"
            "  返回 Nano Banana 提示词参考网站。\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "⚙️ 私聊个人配置：\n"
            "• /设置ai配置 <类型> <api_base> <api_key> <model_name>\n"
            "  类型只能是 gemini、geminichat 或 gpt，模型名必填。\n"
            "• /查看ai配置\n"
            "  查看当前个人类型、地址和模型。\n"
            "示例：\n"
            "/设置ai配置 gemini http://127.0.0.1:7861 pwd gemini-3-pro-image\n"
            "/设置ai配置 geminichat http://127.0.0.1:7861 pwd gemini-3-pro-image\n"
            "/设置ai配置 gpt https://api.openai.com sk-xxx gpt-image-2\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🔌 接口类型：\n"
            "• gemini：Gemini 原生 generateContent 接口\n"
            "• geminichat：OpenAI兼容 /chat/completions 接口\n"
            "• gpt：OpenAI兼容 Images API（生成/编辑）\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "💡 追加参数：\n"
            "生成类指令末尾支持 --参数 值 或 --参数=值。\n"
            "例如：/生图 赛博朋克猫咪 --temperature 0.7 --top_p=0.9 --seed 123\n"
            "帮助别名：/生图帮助"
        )
        yield event.plain_result(help_text)

    @filter.command("生图帮助")
    async def cmd_help_alias(self, event: AstrMessageEvent):
        """帮助（别名）：/生图帮助"""
        async for res in self.cmd_help(event):
            yield res

    @filter.command("海报")
    async def cmd_poster(self, event: AstrMessageEvent):
        """海报（可携带/引用图片）：/海报 [描述]"""
        # 提取全文本输入
        prompt_text = self._get_full_text_input(event, "/海报")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return

        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        default_prompt = (
            "将画面转换为专业电影海报风格，16:9宽屏比例。"
            "采用电影级构图和光影效果，突出主体视觉冲击力。"
            "色彩饱满鲜明，层次分明，具有商业宣传海报的精致质感。"
            "高清细节，专业排版美感，适合用作宣传展示。"
        )
        final_prompt = f"{default_prompt}\n用户补充要求：{prompt}" if prompt else default_prompt

        # 检查是否包含图片
        has_image = self._check_has_image(event)

        # 先返回生成中提示
        yield event.plain_result("🎨 收到请求，正在生成 [海报]...")

        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate", appended_generation_config=appended_params):
            yield res

    @filter.command("表情包")
    async def cmd_sticker(self, event: AstrMessageEvent):
        """表情包（可携带/引用图片）：/表情包 [描述]"""
        # 从消息链抽取完整文本，兼容图文混排
        prompt_text = self._get_full_text_input(event, "/表情包")
        prompt, appended_params, parse_error = self._split_prompt_and_append_params(prompt_text)
        if parse_error:
            yield event.plain_result(parse_error)
            return

        err = self._check_group_access(event)
        if err:
            yield event.plain_result(err)
            return
        allowed, msg = await self._ensure_private_has_config(event)
        if not allowed:
            yield msg
            return
        default_prompt = (
            "将画面转换为Q版可爱表情包风格，LINE贴纸风格。"
            "角色Q版化，大头小身，表情夸张生动有趣。"
            "线条简洁流畅，色彩明快活泼。"
            "背景简单或透明，适合用作聊天表情。"
            "整体风格可爱萌系，富有表现力和感染力。"
        )
        final_prompt = f"{default_prompt}\n用户补充要求：{prompt}" if prompt else default_prompt

        has_image = self._check_has_image(event)

        yield event.plain_result("🎨 收到请求，正在生成 [表情包]...")

        async for res in self.gemini_image_tool(event, image_description=final_prompt, use_reference_images=has_image, mode="edit" if has_image else "generate", appended_generation_config=appended_params):
            yield res

    def _get_reference_image_components(self, event: AstrMessageEvent) -> List[Image]:
        """获取当前消息及引用链中的图片组件，兼容嵌套引用并避免重复。"""
        result: List[Image] = []
        visited = set()

        def walk(chain):
            for component in chain or []:
                component_id = id(component)
                if component_id in visited:
                    continue
                visited.add(component_id)
                if isinstance(component, Image):
                    result.append(component)
                elif isinstance(component, Reply):
                    walk(getattr(component, "chain", None))

        message_obj = getattr(event, "message_obj", None)
        walk(getattr(message_obj, "message", None))
        return result

    def _check_has_image(self, event: AstrMessageEvent) -> bool:
        """检查当前消息或引用消息中是否包含图片组件。"""
        return bool(self._get_reference_image_components(event))

    def _get_full_text_input(self, event: AstrMessageEvent, cmd_prefix: str = "") -> str:
        """
        从消息链中提取所有文本内容，拼接并保留空格/换行，同时移除指令前缀。
        解决 GreedyStr 遇图片截断的问题。
        """
        full_text = ""
        if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
            for component in event.message_obj.message:
                # 只提取纯文本组件，保留其内容（兼容不同版本属性名）
                if isinstance(component, Plain):
                    text = getattr(component, 'text', None) or getattr(component, 'content', '')
                    full_text += text

        # 保留中间空白，去除首尾多余空白
        full_text = full_text.strip()

        # 移除命令前缀
        if cmd_prefix and full_text.startswith(cmd_prefix):
            full_text = full_text[len(cmd_prefix):].strip()

        return full_text

    def _get_quoted_text_input(self, event: AstrMessageEvent) -> str:
        """提取 QQ 引用消息中的文字，兼容 Reply.chain 与 message_str。"""
        message_obj = getattr(event, "message_obj", None)
        root_chain = getattr(message_obj, "message", None) or []
        blocks: List[str] = []
        seen_reply_ids = set()

        def append_block(value: Any):
            text = str(value or "").strip()
            if text and text not in blocks:
                blocks.append(text)

        def read_reply(reply: Reply):
            reply_identity = id(reply)
            if reply_identity in seen_reply_ids:
                return
            seen_reply_ids.add(reply_identity)

            reply_chain = getattr(reply, "chain", None) or []
            plain_parts = []
            nested_replies = []
            for component in reply_chain:
                if isinstance(component, Plain):
                    text = getattr(component, "text", None) or getattr(component, "content", "")
                    if str(text or "").strip():
                        plain_parts.append(str(text).strip())
                elif isinstance(component, Reply):
                    nested_replies.append(component)

            if plain_parts:
                append_block("\n".join(plain_parts))
            else:
                # 部分 OneBot/NapCat 事件没有填充 chain，只在这里保存引用正文。
                append_block(getattr(reply, "message_str", None))

            for nested_reply in nested_replies:
                read_reply(nested_reply)

        for component in root_chain:
            if isinstance(component, Reply):
                read_reply(component)

        return "\n\n".join(blocks)

    def _coerce_cli_param_value(self, raw_value: str) -> Any:
        if raw_value is None:
            return True
        value = str(raw_value).strip()
        lower = value.lower()
        if lower in ("true", "false"):
            return lower == "true"
        if lower in ("null", "none"):
            return None
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except Exception:
                pass
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value

    def _normalize_generation_param_key(self, key: str) -> str:
        k = (key or "").strip()
        if not k:
            return k
        normalized = k.replace("-", "_").lower()
        alias_map = {
            "temperature": "temperature",
            "top_p": "topP",
            "topp": "topP",
            "top_k": "topK",
            "topk": "topK",
            "candidate_count": "candidateCount",
            "candidatecount": "candidateCount",
            "max_output_tokens": "maxOutputTokens",
            "maxoutputtokens": "maxOutputTokens",
            "stop_sequences": "stopSequences",
            "stopsequences": "stopSequences",
            "response_mime_type": "responseMimeType",
            "responsemimetype": "responseMimeType",
            "response_modalities": "responseModalities",
            "responsemodalities": "responseModalities",
            "seed": "seed",
        }
        return alias_map.get(normalized, k)

    def _split_prompt_and_append_params(self, text: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """
        Parse trailing --params in command text.
        Example: /生图 一只猫 --temperature 0.7 --top_p=0.9 --seed 123
        """
        raw = (text or "").strip()
        if not raw:
            return "", {}, None

        try:
            tokens = shlex.split(raw)
        except Exception:
            tokens = raw.split()

        if not tokens:
            return "", {}, None

        # Backtrack from tail: only treat trailing --param block as params.
        j = len(tokens) - 1
        while j >= 0:
            token = tokens[j]
            if token.startswith("--"):
                j -= 1
                continue
            if j - 1 >= 0 and tokens[j - 1].startswith("--"):
                j -= 2
                continue
            break

        params_start = j + 1
        if params_start >= len(tokens):
            return " ".join(tokens), {}, None

        prompt_tokens = tokens[:params_start]
        param_tokens = tokens[params_start:]
        if not param_tokens:
            return " ".join(prompt_tokens), {}, None

        parsed_params: Dict[str, Any] = {}
        idx = 0
        while idx < len(param_tokens):
            tok = param_tokens[idx]
            if not tok.startswith("--"):
                return "", {}, f"参数格式错误：`{tok}`。请将所有 `--参数` 放在命令最后。"

            body = tok[2:]
            if not body:
                return "", {}, "参数格式错误：检测到空参数名，请使用 `--参数名 值` 或 `--参数名=值`。"

            if "=" in body:
                key, value = body.split("=", 1)
            else:
                key = body
                if idx + 1 < len(param_tokens) and not param_tokens[idx + 1].startswith("--"):
                    value = param_tokens[idx + 1]
                    idx += 1
                else:
                    value = "true"

            key = self._normalize_generation_param_key(key)
            if not key:
                return "", {}, "参数格式错误：存在空参数名。"

            parsed_params[key] = self._coerce_cli_param_value(value)
            idx += 1

        return " ".join(prompt_tokens).strip(), parsed_params, None

    @filter.command("设置ai配置")
    async def cmd_set_config(self, event: AstrMessageEvent):
        """设置个人API配置：/设置ai配置 <gemini|geminichat|gpt> <api_base> <api_key> <model_name>"""
        # 检查是否为私聊
        if not self._is_private_chat(event):
            yield event.plain_result("该命令仅支持私聊使用")
            return
        
        # 提取参数
        full_text = self._get_full_text_input(event, "/设置ai配置")
        parts = full_text.split()
        
        if len(parts) < 4:
            yield event.plain_result(
                "❌ 参数不足\n\n"
                "使用方法：\n"
                "/设置ai配置 <gemini|geminichat|gpt> <api_base> <api_key> <model_name>\n\n"
                "例如：\n"
                "/设置ai配置 gemini http://127.0.0.1:7861 your_password gemini-3-pro-image\n"
                "/设置ai配置 geminichat http://127.0.0.1:7861 your_password gemini-3-pro-image\n"
                "/设置ai配置 gpt https://api.openai.com sk-xxx gpt-image-2"
            )
            return

        if parts[0].lower() not in ("gemini", "geminichat", "gpt"):
            yield event.plain_result("❌ 类型必须是 gemini、geminichat 或 gpt")
            return
        provider = self._normalize_provider(parts[0])
        api_base = parts[1]
        api_password = parts[2]
        model_name = parts[3].strip()
        if not model_name:
            yield event.plain_result("❌ model_name 不能为空")
            return
        
        # 简单验证
        if not api_base.startswith("http://") and not api_base.startswith("https://"):
            yield event.plain_result("❌ api_base 必须以 http:// 或 https:// 开头")
            return
        
        # 保存配置
        success = await self._save_user_config(event, provider, api_base, api_password, model_name)
        
        if success:
            yield event.plain_result(
                "✅ 配置保存成功！\n\n"
                f"类型: {provider}\n"
                f"API Base: {api_base}\n"
                f"API Password: {'*' * len(api_password)}\n\n"
                f"Model: {model_name}\n\n"
                "现在您可以在私聊中使用生图功能了"
            )
        else:
            yield event.plain_result("❌ 配置保存失败，请稍后重试")

    @filter.command("查看ai配置")
    async def cmd_view_config(self, event: AstrMessageEvent):
        """查看个人API配置（仅私聊）：/查看ai配置"""
        # 检查是否为私聊
        if not self._is_private_chat(event):
            yield event.plain_result("该命令仅支持私聊使用")
            return
        
        # 获取配置
        user_config = await self._get_user_config(event)
        
        if not user_config:
            yield event.plain_result(
                "❌ 您还没有设置个人配置\n\n"
                "请使用以下命令设置：\n"
                "/设置ai配置 <gemini|geminichat|gpt> <api_base> <api_key> <model_name>\n\n"
                "例如：\n"
                "/设置ai配置 gemini http://127.0.0.1:7861 your_password gemini-3-pro-image"
            )
            return
        
        api_base = user_config.get("api_base", "未设置")
        api_password = user_config.get("api_password", "")
        provider = self._normalize_provider(user_config.get("provider", "gemini"))
        model_name = str(user_config.get("model_name") or "").strip() or "未设置（必填）"
        
        yield event.plain_result(
            "⚙️ 当前个人配置：\n\n"
            f"类型: {provider}\n"
            f"API Base: {api_base}\n"
            f"API Password: {'*' * len(api_password)}\n\n"
            f"Model: {model_name}\n\n"
            "如需修改，请重新使用 /设置ai配置 命令"
        )

    # 已移除 gconf 指令组，配置请在 AstrBot 插件设置中修改
