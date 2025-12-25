from pydantic import BaseModel

class StudentCheckinRequest(BaseModel):
    course_id: int
    student_number: str
    # 将来还要加图片或Token，现在先只做身份校验