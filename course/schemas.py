# course/schemas.py
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional

# 1. 前端创建课程时传的参数
class CourseCreate(BaseModel):
    course_name: str
    semester: str       # 如 "2023-秋季"
    location: str       # 如 "教三-101"

# 2. 返回给前端的完整课程信息
class Course(CourseCreate):
    id: int
    teacher_id: int
    # 【新增】非数据库字段，用于前端展示状态
    status: Optional[str] = "completed" 
    progress: Optional[int] = 100

    class Config:
        from_attributes = True # 允许从数据库模型直接转换
        
class SessionSchedule(BaseModel):
    scheduled_time: datetime # 前端传这就行，格式 ISO 8601