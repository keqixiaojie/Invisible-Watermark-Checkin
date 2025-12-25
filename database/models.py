import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column

# ==========================================
# 【新增】魔法代码：让 Python 能找到上一级目录的 config.py
# ==========================================
# 获取当前脚本所在目录的上一级（即项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
# ==========================================

# 从根目录的 config.py 导入配置
from config import SQLALCHEMY_DATABASE_URL
# 引入我们在 config.py 里写好的配置
# 使用配置里的 URL
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 定义基础类
class Base(DeclarativeBase):
    pass

# ==========================================
# 下面开始定义你要求的 5 张表
# ==========================================

# [1] 用户表 (User) - 存储老师或管理员信息
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False) # 用户名唯一
    password_hash: Mapped[str] = mapped_column(String(100), nullable=False)        # 存加密后的密码
    role: Mapped[str] = mapped_column(String(20), default="teacher")               # teacher 或 admin
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)             # 创建时间自动生成

# [2] 课程表 (Course)
class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    course_name: Mapped[str] = mapped_column(String(100), nullable=False)
    semester: Mapped[str] = mapped_column(String(50))                              # 学期，如 "2023-秋季"
    location: Mapped[str] = mapped_column(String(100))                             # 上课地点
    
    # 外键：关联到 User 表的 id，表示这门课是谁教的
    teacher_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

# [3] 学生花名册表 (Roster) - 记录这门课有哪些学生
class Roster(Base):
    __tablename__ = "rosters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # 外键：关联到 Course 表，表示这条记录属于哪门课
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    
    student_number: Mapped[str] = mapped_column(String(20))  # 学号 (注意是字符串)
    name: Mapped[str] = mapped_column(String(50))            # 学生姓名
    class_name: Mapped[str] = mapped_column(String(50))      # 行政班级，如 "计算机2班"

# [4] 签到场次表 (CheckinSession) - 老师每发起一次签到，就是一条记录
class CheckinSession(Base):
    __tablename__ = "checkin_sessions"
    table_args= {'extend_existing': True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # 外键：属于哪门课的签到
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    
    # [新增] 预约的计划开始时间 (必填)
    scheduled_time: Mapped[datetime] = mapped_column(nullable=False)
    
    # [修改] 实际开始时间 (变为可空，点击“开始”时才写入)
    start_time: Mapped[datetime] = mapped_column(nullable=True)
    end_time: Mapped[datetime] = mapped_column(nullable=True)          # 结束时间
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)     # 状态：是否进行中
    # [新增] 记录本次签到随机选中的是第几组水印 (0, 1, 2)
    code_idx: Mapped[int] = mapped_column(Integer, nullable=True)              # 签到码的索引,用于后续水印验证
    # [新增] 存储水印相关信息的路径 (JSON文件路径)
    # 这个JSON里包含：ignore_mask路径, global_levels数据, 3组图片序列的文件夹路径
    watermark_meta_path: Mapped[str] = mapped_column(String(200), nullable=True)

# [5] 签到记录表 (AttendanceRecord) - 学生每扫一次码，就是一条记录
class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # 外键：属于哪次签到活动
    session_id: Mapped[int] = mapped_column(ForeignKey("checkin_sessions.id"))
    
    student_number: Mapped[str] = mapped_column(String(20)) # 学号
    checkin_time: Mapped[datetime] = mapped_column(default=datetime.now) # 签到时间
    status: Mapped[str] = mapped_column(String(20), default="normal")    # normal(正常), late(迟到), manual(补签)
    device_info: Mapped[str] = mapped_column(String(200), nullable=True) # IP或设备指纹
    


# ==========================================
# 核心执行代码：创建数据库
# ==========================================
if __name__ == "__main__":
    print("正在初始化数据库...")
    print(f"读取的数据库路径: {SQLALCHEMY_DATABASE_URL}")
    Base.metadata.create_all(engine)
    print("数据库表结构创建成功！")