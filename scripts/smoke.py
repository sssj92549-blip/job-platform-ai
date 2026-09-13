"""真实依赖冒烟：使用合成简历，不读业务文件。--live才调用收费DeepSeek接口。"""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pymupdf

from app import schemas as s
from app.config import load_settings
from app.llm import DeepSeekClient
from app.pdf import PdfParser
from app.service import AiService
from app.vector import VectorService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="允许使用DeepSeek进行4次真实生成调用")
    args = parser.parse_args()
    cfg = load_settings()
    with TemporaryDirectory() as temp:
        cfg.storage.uploads_root = Path(temp)
        cfg.storage.chroma_path = Path(temp) / "chroma"
        pdf, vector, llm = PdfParser(cfg), VectorService(cfg), DeepSeekClient(cfg)
        try:
            with pymupdf.open() as doc:
                page = doc.new_page()
                page.insert_text(
                    (50, 80),
                    "姓名：张测试  学历：本科\n技能：Java、Spring Boot、MySQL\n项目经历：开发招聘平台后端接口。",
                    fontname="china-s",
                    fontsize=16,
                )
                doc.save(Path(temp) / "text.pdf")
                picture = page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
            with pymupdf.open() as doc:
                page = doc.new_page()
                page.insert_image(page.rect, stream=picture)
                doc.save(Path(temp) / "scan.pdf")
            assert pdf.extract("text.pdf")["extractionMethod"] == "TEXT"
            scan = pdf.extract("scan.pdf")
            print("OCR synthetic text:", ascii(scan["extractedText"]))
            assert scan["extractionMethod"] == "OCR" and len(scan["extractedText"]) > 20
            print("PDF text + PaddleOCR scan: OK")
            vector.warmup()
            vector.upsert(
                "1",
                1,
                s.VectorRequest(
                    candidateId="1",
                    text="Java Spring Boot MySQL 后端开发",
                    metadata=s.Metadata(confirmed=True, discoverable=True),
                ),
            )
            vector.upsert(
                "2",
                1,
                s.VectorRequest(
                    candidateId="2",
                    text="平面设计 Photoshop 视觉设计",
                    metadata=s.Metadata(confirmed=True, discoverable=False),
                ),
            )
            result = vector.search(
                s.SearchRequest(
                    jobId="1",
                    jobVersion=1,
                    queryText="Java后端开发",
                    topK=10,
                    minSimilarity=0,
                    eligibleResumes=[
                        s.ResumeRef(resumeId="1", resumeVersion=1),
                        s.ResumeRef(resumeId="2", resumeVersion=1),
                    ],
                )
            )
            assert [r["candidateId"] for r in result["matches"]] == ["1"]
            vector.delete("1", 1)
            print("Chinese embedding + Chroma whitelist retrieval: OK")
            if args.live:
                service = AiService(llm, pdf, vector)
                parsed = service.parse_resume(
                    s.ParseRequest(resumeId="1", resumeVersion=1, filePath="scan.pdf")
                )
                assert parsed["parsedSkills"]
                job = s.Job(
                    title="Java后端开发",
                    description="开发业务接口",
                    requirements="熟悉Java及MySQL",
                    skills=["Java", "MySQL"],
                )
                service.match(s.MatchRequest(resumeText=parsed["extractedText"], job=job))
                questions = service.interview_questions(
                    s.InterviewRequest(resumeText=parsed["extractedText"], job=job, count=10)
                )
                assert len(questions["questions"]) == 10
                service.assistant(
                    s.AssistantRequest(
                        question="适合什么岗位？",
                        profile=s.AssistantProfile(
                            name="张测试", education="BACHELOR", skills=["Java"]
                        ),
                    )
                )
                print("DeepSeek parse / match / 10 questions / assistant: OK")
        finally:
            llm.close()
            vector.close()


if __name__ == "__main__":
    main()
