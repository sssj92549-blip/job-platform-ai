# Java → Python 内部接口

## 10. Java → Python 内部接口

所有接口要求 `X-Internal-Token`，JSON 请求同时携带 `X-Request-Id` 便于关联日志。以下无业务 Session；鉴权失败 HTTP 401。Python 不直接更新 MySQL，Java 负责权限、事务、状态和异步任务持久化。

### 10.1 简历解析

`POST /internal/resumes/parse`

输入：

```json
{"resumeId":"3001","resumeVersion":1,"filePath":"resumes/9f3c2a.pdf"}
```

`filePath` 为共享 uploads 根目录下的相对路径。Java、Python 各自配置同一个共享目录（同机可为同一磁盘路径）；Python 规范化路径后检查必须位于根目录内，并拒绝 `..`、绝对路径、符号链接逃逸。不同机器部署时必须先实现文件传输接口，不能沿用不可访问的 Java 本地路径。

输出 data：

```json
{
  "resumeId": "3001", "resumeVersion": 1,
  "extractedText": "简历全文……", "extractionMethod": "MIXED", "pageCount": 2,
  "parsedName": "张三", "parsedPhone": "13900139000",
  "parsedEducation": "BACHELOR", "parsedSkills": ["Java", "MySQL"],
  "parsedSummary": "具有Java项目经验"
}
```

`extractionMethod` 为 TEXT/OCR/MIXED。请求限制与外部上传一致，字段提取失败返回 42201。该接口不直接写 Chroma，Java 成功落库后再调用向量写入。

### 10.2 人岗匹配

`POST /internal/ai/match`

输入：

```json
{
  "resumeText": "已确认资料与简历全文……",
  "job": {"title":"Java开发工程师","description":"参与后端开发","requirements":"熟悉Spring Boot","skills":["Java"]}
}
```

输出 data：`MatchResult`，示例 `{"score":86,"reasons":["具备Java经验"],"gaps":["缺少部署经验"]}`。

### 10.3 面试题生成

`POST /internal/ai/interview-questions`

输入：与匹配接口相同，额外必填 `count: 10`（本期只允许 10）。

输出 data：`InterviewResult`，结构如下，其中 questions 必须实际包含 10 项：

```text
{questions: [{number: 1, question: "如何保证重复投递不会产生多条记录？", direction: "数据库与并发", assessmentPoints: ["唯一约束", "事务处理"]}, ...共10项]}
```

### 10.4 求职助手

`POST /internal/ai/assistant`

输入：

```json
{
  "question": "我适合什么岗位？",
  "profile": {"name":"张三","education":"BACHELOR","skills":["Java","MySQL"],"summary":"有Java项目经验"}
}
```

profile 中 summary 可为 null，其他字段必填。无需传手机号。输出 data：`{"answer":"可以优先考虑Java后端开发实习或初级岗位……"}`。

### 10.5 简历向量写入

`PUT /internal/vector/resumes/{resumeId}/versions/{resumeVersion}`

输入（所有字段必填）：

```json
{
  "candidateId": "1001", "text": "学历、技能、项目经历等检索文本……",
  "metadata": {"confirmed": true, "discoverable": true}
}
```

输出 data：`{resumeId: string, resumeVersion: integer, indexed: true, embeddingModel: string, dimension: integer}`。

按 `resumeId:resumeVersion` 作为 Chroma document ID 幂等 upsert，同版本重试不会重复插入。解析成功时可先写 confirmed=false；用户确认后写新版本并清理旧版本。检索文本由 Java 组装，尽量去除姓名、电话等非匹配字段；确认后以最终资料替换结构化部分。向量模型和维度由配置固定，更换模型需要重建集合，禁止混用。

### 10.6 简历向量删除

`DELETE /internal/vector/resumes/{resumeId}/versions/{resumeVersion}`

输入：路径 resumeId、resumeVersion；无 body。输出 data：`{resumeId: string, resumeVersion: integer, deleted: true}`；不存在也成功。Java 在旧版本、删除、关闭发现、封禁时安排清理对应版本；检索白名单确保迟到的向量写入不会重新暴露无效版本。

### 10.7 人才向量检索

`POST /internal/vector/talents/search`

输入：

```json
{
  "jobId": "2001", "jobVersion": 1,
  "queryText": "Java后端开发，熟悉Spring Boot与MySQL",
  "topK": 10, "minSimilarity": 0.6,
  "eligibleResumes": [{"resumeId":"3003","resumeVersion":2}]
}
```

全部字段必填；eligibleResumes 空数组立即返回空结果，不允许理解为不限制范围。Python 仅在指定版本且 confirmed/discoverable 均为 true 的文档内检索。queryText 最多 25000 字；索引 text、匹配 resumeText 最多 60000 字；向量模型长度限制由 Python 通过分块编码并聚合为单简历向量处理，不能静默丢弃文末内容。

输出 data：

```json
{
  "jobId": "2001", "jobVersion": 1,
  "matches": [{"resumeId":"3003","resumeVersion":2,"candidateId":"1003","similarity":0.87}]
}
```

Java 根据返回 ID 重新校验并读取确认资料，组装外部 Candidate。职位查询向量可按 jobId/jobVersion 缓存，不要求单独持久化职位向量集合。Chroma 或 embedding 服务不可用返回 50301，不能伪装成成功的空推荐。

### 10.8 健康检查

`GET /internal/health`

输入：无（仍需内部认证）。输出 data：`{status: "UP"|"DEGRADED", ocrReady: boolean, vectorReady: boolean, llmConfigured: boolean}`。不实际调用付费模型；llmConfigured 只说明配置存在。依赖不可用 HTTP 503，code=50301、data=null，message 描述不可用依赖。

