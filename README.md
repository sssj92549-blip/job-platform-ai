# job-platform-ai

招聘平台独立 Python AI 服务：FastAPI + DeepSeek + PyMuPDF/PaddleOCR + 中文 Embedding + Chroma。只实现既定范围，不引入 LangGraph、完整 RAG、聊天历史或 MySQL 业务写入。接口路径无版本前缀。

## 配置与启动

推荐 Python 3.10，支持 3.10～3.12；当前开发机已创建 `.venv`。依赖版本固定在 `uv.lock`。

```powershell
# 首次安装（如果没有uv）
python -m pip install uv
python -m uv sync --frozen --python 3.10
# 首次配置：仅当本地尚无local.yml时执行，避免覆盖已有密钥
Copy-Item local.example.yml local.yml
# 填好local.yml后启动
.venv\Scripts\python.exe run.py
```

默认监听 `127.0.0.1:7999`，首次启动下载OCR和Embedding模型，后续复用缓存。启动预热可能需要几分钟，失败后健康检查明确返回503，不伪装成可用。只运行一个worker，避免重复加载模型或争用本地Chroma。

| 文件 | 用途 |
|---|---|
| `default.yml` | DeepSeek模型名、Embedding模型名、OCR模型名及公共默认配置，可提交 |
| `local.yml` | API Key、内部认证token、当前机器uploads路径，Git忽略 |
| `local.example.yml` | 无真实密钥的配置模板，可提交 |

配置加载顺序：default.yml → local.yml，相对路径以项目根目录为准。不要把密钥写回default.yml。当前DeepSeek模型为 `deepseek-flash`；本地向量模型为 `BAAI/bge-small-zh-v1.5`，512维，通过FastEmbed/ONNX在CPU上运行。DeepSeek负责理解和生成，向量编码由本地模型负责。

## 分层

```text
app/main.py       FastAPI路由、内部认证、统一信封、异常转换
app/service.py    AiService：所有AI业务入口集中在一个类
app/llm.py        DeepSeekClient：调用、超时处理、JSON结构校验
app/pdf.py        PdfParser：安全路径、PDF读取、逐页OCR降级
app/vector.py     VectorService：全文分块编码、Chroma持久化与检索
app/schemas.py    Pydantic输入和结果约束
app/config.py     YAML配置加载
tests/            无付费调用的自动化测试
scripts/smoke.py  真实OCR/向量验证，--live额外调用DeepSeek
```

## 接口

所有接口必带 `X-Internal-Token`，值为local.yml的 `server.internal_token`。Java可附带 `X-Request-Id`，服务回传相同追踪ID；不对浏览器直接开放。

| 方法 | 路径 | 输入 | 成功data |
|---|---|---|---|
| GET | `/internal/health` | 无 | `{status,ocrReady,vectorReady,llmConfigured}` |
| POST | `/internal/resumes/parse` | `{resumeId,resumeVersion,filePath}` | ID/版本、extractedText、extractionMethod、pageCount及parsedName/Phone/Education/Skills/Summary |
| POST | `/internal/ai/match` | `{resumeText,job:{title,description,requirements,skills}}` | `{score,reasons,gaps}` |
| POST | `/internal/ai/interview-questions` | 匹配输入 + `count:10` | `{questions:[{number,question,direction,assessmentPoints}]}`，恰好10题 |
| POST | `/internal/ai/assistant` | `{question,profile:{name,education,skills,summary?}}` | `{answer}` |
| PUT | `/internal/vector/resumes/{resumeId}/versions/{resumeVersion}` | `{candidateId,text,metadata:{confirmed,discoverable}}` | `{resumeId,resumeVersion,indexed,embeddingModel,dimension}` |
| DELETE | `/internal/vector/resumes/{resumeId}/versions/{resumeVersion}` | 无 | `{resumeId,resumeVersion,deleted}`，不存在也成功 |
| POST | `/internal/vector/talents/search` | `{jobId,jobVersion,queryText,topK,minSimilarity,eligibleResumes:[{resumeId,resumeVersion}]}` | `{jobId,jobVersion,matches:[{resumeId,resumeVersion,candidateId,similarity}]}` |

统一信封：`{code:0,message:"success",data:...,requestId:"..."}`。失败data为null，HTTP状态和业务码分别为400/40001、401/40101、404/40401、413/41301、422/42201、502/50201、503/50301、504/50401。DeepSeek超时返回504；余额不足、密钥无效等上游失败返回502，不输出上游私密响应。

完整Java/Python契约见 `docs/internal-api.md`。ID为字符串，版本为正整数。PDF限10MB、20页，全文限60000字；拒绝加密文件、绝对路径、父目录跳转和符号链接越界。文本页直接提取，扫描页或较大图片混合页用OCR；识别结果仍需Java前端确认，不能自动当作真实资料。

## Java对接

Java端 `app.ai.base-url` 使用 `http://127.0.0.1:7999`，`app.ai.internal-token` 与Python的 `server.internal_token` 保持一致。当前Java基础框架的RestTemplate会自动带上该内部认证头；后续业务Service仍需实现HTTP调用、事务和异步任务持久化。

两端uploads必须指向同一目录，本机为 `E:/java/code/jobPlatform/uploads`。请求示例：

```json
{"resumeId":"3001","resumeVersion":1,"filePath":"resumes/abc.pdf"}
```

不要传 `E:/...` 绝对路径。Python不修改MySQL，也不自动给求职者更新个人档案。解析成功后Java保存结果，并调用向量写入接口；确认后Java传新版本和confirmed=true。

生成接口在Python侧同步返回，Java负责异步任务和前端轮询；每次请求独立，不保存多轮聊天历史。DeepSeek请求默认90秒读取超时、3秒连接超时，不自动重试收费调用。

## 向量检索规则

- `resumeId:resumeVersion` 为文档ID，upsert幂等。只保存向量和必要元数据，不重复存储全文。
- 中文文本逐段保留全部字符，再校验真实token数，分块向量按字符数量加权平均并归一化，不静默截断长简历。
- 严格按eligibleResumes限定版本，且confirmed、discoverable都为true；空白名单立即返回空数组。Java仍需排除已投递候选人，并对结果二次检查权限。
- 使用余弦距离，similarity为clamp(1-distance,0,1)，不是DeepSeek的0～100匹配评分。
- 模型名、分块方式、维度记录到集合签名中，配置不一致拒绝使用旧集合。更换Embedding模型应改集合名并由Java重新索引。
- Chroma数据在data/chroma，Embedding缓存位于data/models；PaddleOCR缓存由PaddleX管理，Windows通常位于用户目录 `.paddlex/official_models`。模型和向量文件不上传Git。

## 验证

```powershell
.venv\Scripts\pytest.exe -q
.venv\Scripts\ruff.exe check app tests scripts run.py
# 真实OCR、Embedding和Chroma验证，不调用收费模型
.venv\Scripts\python.exe scripts/smoke.py
# 额外4次DeepSeek调用：解析、匹配、面试题、助手
.venv\Scripts\python.exe scripts/smoke.py --live
```

自动化测试覆盖认证、请求追踪、参数限制、PDF路径安全、扫描降级、加密PDF拒绝、非法模型输出、超时、真实Chroma白名单/可见性过滤及全文分块。测试替身仅在tests目录中，生产服务没有假结果回退。

官方资料：[DeepSeek API](https://api-docs.deepseek.com/)、[PaddleOCR](https://www.paddleocr.ai/latest/en/quick_start.html)、[FastEmbed中文模型](https://qdrant.github.io/fastembed/examples/Supported_Models/)、[Chroma集合配置](https://docs.trychroma.com/docs/collections/configure)。
