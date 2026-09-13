"""与Java契约对应的请求/结果；拒绝多余字段与越界模型输出。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

Id = Annotated[str, Field(pattern=r"^[1-9][0-9]{0,18}$")]
Version = Annotated[int, Field(strict=True, ge=1)]
Short = Annotated[str, Field(min_length=1, max_length=100)]
Education = Literal["HIGH_SCHOOL", "JUNIOR_COLLEGE", "BACHELOR", "MASTER", "DOCTOR", "OTHER"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParseRequest(Model):
    resumeId: Id
    resumeVersion: Version
    filePath: str = Field(min_length=1, max_length=255)


class Parsed(Model):
    parsedName: str | None = Field(max_length=50)
    parsedPhone: str | None = Field(max_length=32)
    parsedEducation: Education | None
    parsedSkills: list[Short] = Field(max_length=50)
    parsedSummary: str | None = Field(max_length=2000)


class Job(Model):
    title: str = Field(min_length=2, max_length=100)
    description: str = Field(min_length=1, max_length=10000)
    requirements: str = Field(min_length=1, max_length=10000)
    skills: list[Short] = Field(max_length=50)


class MatchRequest(Model):
    resumeText: str = Field(min_length=1, max_length=60000)
    job: Job


class InterviewRequest(MatchRequest):
    count: Annotated[StrictInt, Field(ge=10, le=10)]


class MatchResult(Model):
    score: Annotated[StrictInt, Field(ge=0, le=100)]
    reasons: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        min_length=1, max_length=20
    )
    gaps: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(max_length=20)


class Question(Model):
    number: Annotated[StrictInt, Field(ge=1, le=10)]
    question: str = Field(min_length=1, max_length=2000)
    direction: str = Field(min_length=1, max_length=200)
    assessmentPoints: list[Short] = Field(min_length=1, max_length=20)


class InterviewResult(Model):
    questions: list[Question] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def unique_numbers(self):
        if [q.number for q in self.questions] != list(range(1, 11)):
            raise ValueError("题号必须按1至10排列")
        return self


class AssistantProfile(Model):
    name: str = Field(min_length=1, max_length=50)
    education: Education
    skills: list[Short] = Field(max_length=50)
    summary: str | None = Field(default=None, max_length=2000)


class AssistantRequest(Model):
    question: str = Field(min_length=1, max_length=2000)
    profile: AssistantProfile


class AssistantResult(Model):
    answer: str = Field(min_length=1, max_length=10000)


class Metadata(Model):
    confirmed: StrictBool
    discoverable: StrictBool


class VectorRequest(Model):
    candidateId: Id
    text: str = Field(min_length=1, max_length=60000)
    metadata: Metadata


class ResumeRef(Model):
    resumeId: Id
    resumeVersion: Version


class SearchRequest(Model):
    jobId: Id
    jobVersion: Version
    queryText: str = Field(min_length=1, max_length=25000)
    topK: Annotated[StrictInt, Field(ge=1, le=50)]
    minSimilarity: float = Field(ge=0, le=1, allow_inf_nan=False)
    eligibleResumes: list[ResumeRef] = Field(max_length=10000)
