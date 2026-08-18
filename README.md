## Gemini / GPT Image Plugin

- 标识名：`gemini-image`
- 功能：支持 Gemini Nano Banana Pro 与 OpenAI GPT Image 2 生图/改图，自动发送到 QQ（Napcat）。
- 设计：模块化/可扩展，解耦 API 客户端，与 AstrBot 交互通过指令组。

### 安装与配置

1. 将本目录放入 AstrBot 插件目录。
2. 安装 `requirements.txt` 中的依赖（AstrBot 从插件市场安装时通常会自动处理）。
3. 在 AstrBot 后台或配置文件中为该插件配置：

```json
{
  "provider": "gemini",
  "gcli2api_base_url": "http://127.0.0.1:7861",
  "gcli2api_api_password": "pwd",
  "model_name": "gemini-3-pro-image",
  "max_retry_attempts": 3,
  "nap_file_forward_enabled": false,
  "nap_server_address": "",
  "nap_server_port": 0
}
```

- `provider`: 可填写 `gemini`、`geminichat` 或 `gpt`。
- `model_name`: 必填，插件会原样使用该名称，不自动推断或替换。请按实际服务商填写，例如 `gemini-3-pro-image`、`gpt-image-2` 或代理自定义模型名。
- `gcli2api_base_url`: 为兼容旧配置保留该名称；GPT 模式可填写 `https://api.openai.com` 或兼容代理地址。
- `gcli2api_api_password`: Gemini 模式填写 API Key/代理密码，GPT 模式填写 OpenAI API Key。

### 指令

- `/生图 [提示词]`：支持直接输入提示词或引用 QQ 文字消息；引用后在命令中输入的文字会作为补充要求，不会读取参考图片。
- `/改图 <提示词>`：基于消息中携带/引用的图片进行改图。
- `/手办化`：携带/引用图片后，使用内置提示词进行“手办化”改图。
- `/coser化 [补充要求]`：携带/引用角色插画，生成真人 Coser 摄影图。
- `/生成角色设定 [补充要求]`：携带/引用角色图，生成比例、三视图、表情、动作和服装设定。
- `/文章信息图 [文章内容]`：提炼文章核心信息并生成全部使用简体中文文字的信息图；支持直接输入、引用 QQ 文字消息，或在引用内容后继续附加文字，也可携带参考图。
- `/提示词参考`：返回 Nano Banana 提示词参考网站。
- `/aiimg帮助`：查看用法说明。
- `/设置ai配置 <gemini|geminichat|gpt> <api_base> <api_key> <model_name>`：在私聊中设置个人模型类型与 API，模型名必填。
- `/查看ai配置`：查看个人配置。

个人配置示例：

```text
/设置ai配置 gemini http://127.0.0.1:7861 your_password gemini-3-pro-image
/设置ai配置 geminichat http://127.0.0.1:7861 your_password gemini-3-pro-image
/设置ai配置 gpt https://api.openai.com sk-xxx gpt-image-2
```

新增创意指令的提示词参考自 [Nano Banana 提示词合集](https://github.com/newaiproxy/nanobanana-prompt)，并针对插件的生图/改图流程进行了扩展。

### 管控与限流

- 群白/黑名单（插件设置）：
  - `group_control_mode`: `off`/`whitelist`/`blacklist`
  - `group_list`: 群号列表（字符串）
- 群限流（插件设置）：
  - `group_rate_window_seconds`: 限流周期（秒）
  - `group_rate_max_calls`: 每群每周期最大调用次数

说明：
- 私聊不受白/黑名单与群限流影响。
- 限流为运行时内存级计数，重启后会重置（如需持久化可再议）。
  （配置修改请在 AstrBot 插件设置中进行，无需命令）

### 发送到 QQ

- 优先使用 `callback_api_base`（AstrBot 全局配置）生成临时下载链接；失败则回退到本地文件发送。
- AstrBot 与 NapCat 如果把宿主机同一个目录挂载为相同的 `/AstrBot/data`，请保持 `nap_file_forward_enabled: false`，NapCat 可直接读取生成图片路径。
- `nap_file_forward_enabled` 只用于另行部署了本插件配套 TCP 文件接收服务的环境；3658 不是 NapCat 自带的服务端口。
- 如果确实启用中转，Docker 内不能用 `localhost` 指代另一个容器，应填写目标服务名，并确保目标容器确实监听对应端口。

### API 对接

Gemini：

- 端点：`/v1beta/models/{model}:generateContent`（非流式），`/v1beta/models/{model}:streamGenerateContent`（流式，默认附加 `?alt=sse`）。
- 鉴权：若配置了 `gcli2api_api_password`，将使用请求头 `x-goog-api-key: <password>`；也可通过 URL `?key=`（插件默认用请求头）。
- 负载：`contents=[{role:user, parts:[{text}, {inlineData}...]}]`，与官方 SDK 示例一致；改图时将用户图片转为 inlineData 放入 parts。

Gemini Chat（第三方 OpenAI 兼容网关）：

- 类型填写 `geminichat`，模型名必须按网关实际提供的名称填写。
- 端点：`/v1/chat/completions`；如果 API Base 已经以 `/v1` 结尾，不会重复拼接。
- 请求使用 `messages`、多模态 `image_url` 内容块以及 `modalities: ["text", "image"]`。
- 兼容从 `choices[0].message.content`、`content[]`、`message.images[]`、`choice.images[]` 和顶层 `images[]` 读取 data URL 或 HTTP 图片。
- 如果兼容网关明确拒绝 `modalities` 参数，会自动移除该参数重试一次。

GPT：

- 生图端点：`/v1/images/generations`，JSON 请求。
- 改图端点：`/v1/images/edits`，参考图通过 multipart `image[]` 上传。
- `quality` 默认使用 `high`；仍可通过指令末尾的 `--quality low|medium|high|auto` 显式覆盖。
- 响应：读取 `data[0].b64_json`；兼容返回图片 URL 的代理服务。
- GPT Image 2 不复用 Gemini SSE，使用非流式 Images API。

### 设计说明

- 遵循 AstrBot 插件规范：`metadata.yaml` + `@register` + `filter.command`。
- 扩展性：三个 API 客户端分别封装 Gemini 原生、Gemini Chat 兼容和 OpenAI Images 协议。
- 解耦：业务逻辑与网络请求分离，专注 gcli2api 转发与 AstrBot 交互。
- 开闭原则：新增模型/路径仅需修改配置或替换 API 客户端，无需改动指令/对外接口。

### 注意事项

- Gemini 模式请确保 Google 凭证或 gcli2api 可用；GPT 模式请确保 OpenAI/兼容代理支持 Images API。
- 参考图会根据实际文件头识别格式，而不是相信文件扩展名或 Data URI 标签；GIF/BMP/TIFF 等格式会自动转为 PNG，动画图使用首帧。
- `/改图`、`/手办化`、`/coser化` 和 `/生成角色设定` 在参考图读取失败时会停止并报错，不会静默降级成无参考图的普通生图。
