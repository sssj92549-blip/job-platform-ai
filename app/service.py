"""统一AI业务入口，HTTP路由不包含提示词或推理流程。"""

from . import schemas as s


class AiService:
    def __init__(self, llm, pdf, vector):
        self.llm, self.pdf, self.vector = llm, pdf, vector

    def parse_resume(self, body: s.ParseRequest) -> dict:
        """提取PDF文本，再调用DeepSeek生成结构化简历，不写MySQL。"""
        extracted = self.pdf.extract(body.filePath)
        parsed = self.llm.generate(
            "提取姓名、电话、学历、技能和摘要。缺失标量填null，技能缺失填[]。",
            {"resumeText": extracted["extractedText"]},
            s.Parsed,
        )
        return {
            "resumeId": body.resumeId,
            "resumeVersion": body.resumeVersion,
            **extracted,
            **parsed,
        }

    def match(self, body: s.MatchRequest) -> dict:
        return self.llm.generate(
            "依据技能、经历与职位要求给出整数0至100评分、匹配原因和差距。",
            body.model_dump(),
            s.MatchResult,
        )

    def interview_questions(self, body: s.InterviewRequest) -> dict:
        return self.llm.generate(
            "生成恰好10道个性化面试题，题号从1到10，包含提问方向和考察点。",
            body.model_dump(),
            s.InterviewResult,
        )

    def assistant(self, body: s.AssistantRequest) -> dict:
        return self.llm.generate(
            "基于给定简历回答此次求职问题。不虚构用户经历或岗位空缺。",
            body.model_dump(),
            s.AssistantResult,
        )

    def upsert_resume(self, resume_id: str, version: int, body: s.VectorRequest) -> dict:
        return self.vector.upsert(resume_id, version, body)

    def delete_resume(self, resume_id: str, version: int) -> dict:
        return self.vector.delete(resume_id, version)

    def search_talents(self, body: s.SearchRequest) -> dict:
        return self.vector.search(body)
