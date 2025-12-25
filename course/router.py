# course/router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from . import schemas
from database.models import Course
from dependencies import get_db, get_current_user # 导入刚才升级的依赖

# 1. 新增导入
import pandas as pd
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File # <--- 加上 UploadFile, File
from database.models import Course, Roster # <--- 记得导入 Roster 模型

import os
import shutil

import json
import random
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks # <--- 引入 BackgroundTasks

from sqlalchemy.orm import Session
from typing import List

from . import schemas
from database.models import Course, Roster, CheckinSession, User
from dependencies import get_db, get_current_user
from config import BASE_DIR

# 引入刚才写的后台任务函数
# 引入任务和全局变量
from utils.tasks import generate_course_watermarks, TASK_PROGRESS

from pydantic import BaseModel

# 记得在文件头部导入 AttendanceRecord 模型
from database.models import AttendanceRecord, Roster, CheckinSession

from fastapi.responses import StreamingResponse # <--- 用于下载文件
from sqlalchemy import desc # <--- 用于按时间倒序排列
from datetime import datetime, timedelta

from fastapi import Request

router = APIRouter(
    prefix="/courses",
    tags=["课程管理 (Course)"]
)
# course/router.py


# ===========================
# 接口: 获取课程列表 (注入进度状态)
# ===========================
@router.get("/", response_model=List[schemas.Course])
def read_courses(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 查数据库
    courses = db.query(Course).filter(Course.teacher_id == current_user.id).all()
    
    results = []
    for c in courses:
        # 转为字典以便修改 (SQLAlchemy对象直接改属性可能会报错)
        c_dict = {
            "id": c.id,
            "course_name": c.course_name,
            "semester": c.semester,
            "location": c.location,
            "teacher_id": c.teacher_id,
            "status": "completed", # 默认
            "progress": 100
        }

        # 2. 检查生成状态
        # 优先查内存中的正在运行任务
        if c.id in TASK_PROGRESS:
            task_info = TASK_PROGRESS[c.id]
            c_dict["status"] = task_info["status"]
            c_dict["progress"] = task_info["progress"]
        else:
            # 内存里没有，检查磁盘上有没有 metadata.json
            meta_path = BASE_DIR / "static" / "courses" / str(c.id) / "metadata.json"
            if meta_path.exists():
                c_dict["status"] = "completed"
                c_dict["progress"] = 100
            else:
                # 既没在跑，也没文件，说明是 Pending (刚创建还没跑) 或 Error
                # 这里简单处理为 pending
                c_dict["status"] = "pending"
                c_dict["progress"] = 0
        
        results.append(c_dict)

    return results

# ... create_course 接口不需要大改，保持之前那种“触发后台任务后立即返回”的逻辑即可 ...
# ===========================
# 接口 2: 创建新课程 (修改版 - 增加查重)
# ===========================
@router.post("/", response_model=schemas.Course)
def create_course(
    request: Request,
    course: schemas.CourseCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 【新增】查重逻辑
    exists = db.query(Course).filter(
        Course.teacher_id == current_user.id,
        Course.course_name == course.course_name,
        Course.semester == course.semester # 同名但不同学期是可以的
    ).first()
    
    if exists:
        raise HTTPException(status_code=400, detail=f"课程【{course.course_name}】在该学期已存在")

    # 2. 写入数据库
    db_course = Course(
        **course.dict(),
        teacher_id=current_user.id 
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    
    # 3. 【核心】获取当前访问的域名
    # request.base_url 通常返回 "http://127.0.0.1:8000/" (带末尾斜杠)
    # 我们转为字符串并去掉末尾的斜杠
    current_domain = str(request.base_url).rstrip("/")
    
    print(f"[Debug] 检测到当前域名: {current_domain}")
    # 3. 触发后台任务
    background_tasks.add_task(generate_course_watermarks, db_course.id, current_domain)
    
    return db_course

# ===========================
# [新增] 接口: 获取任务进度 (供前端轮询)
# ===========================
@router.get("/{course_id}/task_progress")
def get_task_progress(course_id: int):
    # 1. 先查内存里的进度
    progress = TASK_PROGRESS.get(course_id)
    
    if progress:
        return progress
    
    # 2. 如果内存里没有（可能是服务器重启了，或者任务还没开始）
    # 检查一下 metadata.json 是否存在，存在则说明已经完成了
    meta_path = BASE_DIR / "static" / "courses" / str(course_id) / "metadata.json"
    if meta_path.exists():
        return {"status": "completed", "progress": 100, "message": "已完成"}
    
    # 3. 既没在运行也没文件，说明还没开始
    return {"status": "pending", "progress": 0, "message": "等待队列中..."}
    

# ===========================
# 接口 3: 删除课程 (DELETE)
# ===========================
@router.delete("/{course_id}")
def delete_course(
    course_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 查找课程，并确保是当前老师的课
    course = db.query(Course).filter(
        Course.id == course_id, 
        Course.teacher_id == current_user.id
    ).first()
    
    if course is None:
        raise HTTPException(status_code=404, detail="课程不存在或无权删除")
    
    db.delete(course)
    db.commit()
    return {"msg": "删除成功"}

# ===========================
# 接口 4: 导入学生名单 (Excel)
# ===========================
@router.post("/{course_id}/import")
async def import_roster(
    course_id: int,
    file: UploadFile = File(...), # 接收文件
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 权限校验：先看看这门课是不是这个老师的
    course = db.query(Course).filter(Course.id == course_id, Course.teacher_id == current_user.id).first()
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在或无权操作")

    # 2. 读取 Excel 文件
    # 只要是 .xlsx 结尾的都可以
    try:
        contents = await file.read()
        # 使用 pandas 读取二进制数据
        df = pd.read_excel(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")

    # 3. 检查 Excel 列名对不对
    # 我们约定 Excel 必须包含这三列：'学号', '姓名', '班级'
    required_columns = ['学号', '姓名', '班级']
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(status_code=400, detail="Excel格式错误，请包含：学号、姓名、班级")

    # 4. 循环写入数据库
    count = 0
    for index, row in df.iterrows():
        student_no = str(row['学号']) # 转成字符串防止是数字
        name = row['姓名']
        class_name = row['班级']

        # 4.1 防止重复添加：先查一下这个学生是不是已经在名单里了
        exists = db.query(Roster).filter(
            Roster.course_id == course_id,
            Roster.student_number == student_no
        ).first()

        if not exists:
            new_student = Roster(
                course_id=course_id,
                student_number=student_no,
                name=name,
                class_name=class_name
            )
            db.add(new_student)
            count += 1
    
    db.commit()
    return {"msg": f"成功导入 {count} 名学生", "total_rows": len(df)}

# ===========================
# 接口 5: 获取某课程的学生名单
# ===========================
@router.get("/{course_id}/students")
def get_course_students(
    course_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 同样要做权限校验
    course = db.query(Course).filter(Course.id == course_id, Course.teacher_id == current_user.id).first()
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在")
    
    # 查询 Roster 表
    students = db.query(Roster).filter(Roster.course_id == course_id).all()
    return students


    
# ===========================

# 接口 A: 预约签到 (替代原来的 start)

# ===========================

@router.post("/{course_id}/sessions/schedule")
def schedule_session(
    course_id: int,
    schedule_data: schemas.SessionSchedule, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 校验课程
    course = db.query(Course).filter(
        Course.id == course_id, 
        Course.teacher_id == current_user.id
    ).first()
    
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在")

    # 2. 检查元数据路径 (增加调试信息)
    # 确保 BASE_DIR 是绝对路径，更稳健
    abs_base = BASE_DIR.resolve()
    course_meta_path = abs_base / "static" / "courses" / str(course_id) / "metadata.json"
    
    print(f"[Debug] Trying to read metadata from: {course_meta_path}")
    print(f"[Debug] Exists? {course_meta_path.exists()}")

    if not course_meta_path.exists():
        # 如果找不到，尝试打印一下父目录里有啥，方便排查
        parent_dir = course_meta_path.parent
        if parent_dir.exists():
            print(f"[Debug] Parent dir content: {list(parent_dir.iterdir())}")
        else:
            print(f"[Debug] Parent dir {parent_dir} does not exist.")
            
        raise HTTPException(status_code=400, detail=f"资源未就绪，找不到: {course_meta_path}")

    # 3. 读取元数据
    try:
        with open(course_meta_path, 'r', encoding='utf-8') as f: # 显式指定 utf-8
            meta_data = json.load(f)
            groups = meta_data.get('groups', [])
            total_groups = len(groups)
            if total_groups == 0: 
                raise Exception("元数据中没有图片组信息")
    except Exception as e:
        print(f"[Debug] JSON load error: {e}")
        raise HTTPException(status_code=500, detail=f"元数据损坏: {str(e)}")

    # 4. 随机选一组
    selected_group_index = random.randint(0, total_groups - 1)
    print(f"[Debug] Selected group index: {selected_group_index}")

    # 5. 创建 Session
    new_session = CheckinSession(
        course_id=course_id,
        scheduled_time=schedule_data.scheduled_time,
        start_time=None,
        is_active=False,
        code_idx=selected_group_index, 
        watermark_meta_path=str(course_meta_path)
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return {"msg": "预约成功", "session_id": new_session.id}


# ===========================

# 接口 B: 激活签到 (老师点击大屏“开始”时调用)

# ===========================

@router.post("/sessions/{session_id}/activate")

def activate_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    session = db.query(CheckinSession).filter(CheckinSession.id == session_id).first()
    if not session: raise HTTPException(404, "未找到")

    # 1. 把同课程其他的活跃 Session 关掉 (互斥)
    active_s = db.query(CheckinSession).filter(
        CheckinSession.course_id == session.course_id,
        CheckinSession.is_active == True
    ).first()
    if active_s: active_s.is_active = False

    # 2. 激活当前 Session
    session.is_active = True
    session.start_time = datetime.now() # 【关键】验证算法的基准时间以这一刻为准

    db.commit()

    # 3. 【核心补全】返回图片组
    qr_urls = []
    
    # 读取 meta 文件路径 (从数据库字段拿，或者拼路径)
    meta_path = Path(session.watermark_meta_path)
    
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            # 获取所有组
            all_groups = meta.get('groups', [])
            
            # 使用预约时存好的索引 (code_idx)
            idx = session.code_idx
            
            # 防御性编程：万一索引越界（比如重新生成了水印导致组数变少），默认取第0组
            if idx is not None and 0 <= idx < len(all_groups):
                # 这里的 all_groups 结构是: 
                # [ 
                #   { "group_id": 0, "images": ["/static/...", "/static/..."] }, 
                #   { ... } 
                # ]
                # 还是直接是: [ ["url1", "url2"], ["url3"...] ] ?
                # 取决于你 tasks.py 怎么存的。
                # 按照你给的 tasks.py 代码：
                # groups_config.append(group_files) -> 这是一个 URL 列表
                # all_groups_info.append(group_config) -> 这是一个字典列表
                # master_meta["groups"] = all_groups_info -> 这是一个字典列表
                
                # 所以我们要取里面的 "images" 字段
                group_data = all_groups[idx]
                qr_urls = group_data.get("images", [])
                
            elif len(all_groups) > 0:
                # 兜底
                qr_urls = all_groups[0].get("images", [])
                
        except Exception as e:
            print(f"读取图片失败: {e}")
            raise HTTPException(500, "读取水印配置失败")

    return {"msg": "签到已开始", "qr_urls": qr_urls}

# ===========================
# [新增] 接口: 获取当前活跃 Session (供学生端使用)
# ===========================

@router.get("/{course_id}/active_session")
def get_active_session(
    course_id: int,
    db: Session = Depends(get_db)
):
    # 1. 查活跃 Session
    session = db.query(CheckinSession).filter(
        CheckinSession.course_id == course_id,
        CheckinSession.is_active == True
    ).first()
    
    if not session:
        return {"active": False}
    
    # 2. 【新增】为了恢复显示，我们需要读取 metadata.json 拿回图片链接
    # 路径规则要和生成时保持一致
    meta_path = BASE_DIR / "static" / "courses" / str(course_id) / "metadata.json"
    
    qr_urls = []
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                # 简单起见，我们恢复时随机选一组，或者默认选第一组
                # 只要是同一批次生成的，效力是一样的
                groups = meta.get("groups", [])
                if groups:
                    qr_urls = groups[0] 
        except Exception:
            print("读取元数据失败")

    return {
        "active": True,
        "session_id": session.id,
        "qr_urls": qr_urls # <--- 把图片列表也还给前端
    }

# course/router.py

# ... (其他的导入和接口)

# ===========================
# [补回] 接口: 立即开始签到 (Quick Start)
# 用于大屏页面自动创建新会话，或者老师点击“立即开始”
# ===========================
@router.post("/{course_id}/sessions/start")
def start_checkin_session_now(
    course_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 权限校验
    course = db.query(Course).filter(
        Course.id == course_id, 
        Course.teacher_id == current_user.id
    ).first()
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在")

    # 2. 检查资源 (注意：这里使用修正后的路径，不带 /meta)
    course_meta_path = BASE_DIR / "static" / "courses" / str(course_id) / "metadata.json"
    
    if not course_meta_path.exists():
        raise HTTPException(400, "课程资源未生成，请先在列表页等待生成完成")

    # 3. 读取元数据并随机选组
    try:
        with open(course_meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
            groups = meta_data.get('groups', [])
            if not groups: raise Exception("无图片组")
            
            # 随机选一组
            selected_group_idx = random.randint(0, len(groups) - 1)
            selected_group = groups[selected_group_idx]
            qr_urls = selected_group.get("images", [])
    except Exception as e:
        print(f"Meta read error: {e}")
        raise HTTPException(500, "元数据读取失败")

    # 4. 互斥逻辑：关掉该课程其他活跃 Session
    active_s = db.query(CheckinSession).filter(
        CheckinSession.course_id == course_id,
        CheckinSession.is_active == True
    ).first()
    if active_s: active_s.is_active = False

    # 5. 创建并激活 Session
    new_session = CheckinSession(
        course_id=course_id,
        scheduled_time=datetime.now(), # 既然是立即开始，预约时间就是现在
        start_time=datetime.now(),     # 立即激活
        is_active=True,                # 状态：活跃
        code_idx=selected_group_idx,   # 记录选了哪组
        watermark_meta_path=str(course_meta_path)
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return {
        "msg": "签到已开启",
        "session_id": new_session.id,
        "qr_urls": qr_urls
    }
# ===========================
# 接口: 获取签到统计数据 (轮询用)
# ===========================
@router.get("/sessions/{session_id}/stats")
def get_session_stats(
    session_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 查 Session 信息
    session = db.query(CheckinSession).filter(CheckinSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session不存在")

    # 2. 查应到总人数 (根据课程ID查花名册)
    total_count = db.query(Roster).filter(Roster.course_id == session.course_id).count()

    # 3. 查实到人数 (根据 SessionID 查签到记录)
    checked_count = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).count()

    # 4. (可选) 获取最近签到的 3 个人名，用于大屏弹幕效果
    latest_records = db.query(AttendanceRecord, Roster.name)\
        .join(Roster, AttendanceRecord.student_number == Roster.student_number)\
        .filter(AttendanceRecord.session_id == session_id)\
        .order_by(AttendanceRecord.checkin_time.desc())\
        .limit(3).all()
    
    latest_names = [name for _, name in latest_records]

    return {
        "total": total_count,
        "checked": checked_count,
        "latest_names": latest_names
    }
    
    # ==========================================
# 接口 6: 获取某次签到的完整名单 (含未签到的人)
# ==========================================

# course/router.py

from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import pandas as pd
import io

# ... (其他的导入)

# ==========================================
# 接口 6: 获取某次签到的详细名单 (含缺勤)
# ==========================================
@router.get("/sessions/{session_id}/detail")
def get_session_detail(
    session_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 获取 Session 信息
    session = db.query(CheckinSession).filter(CheckinSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session不存在")
    
    # 2. 获取该课程所有学生 (花名册)
    all_students = db.query(Roster).filter(Roster.course_id == session.course_id).all()
    
    # 3. 获取已签到的记录，转成字典方便查询 { "学号": "状态" }
    records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()
    record_map = {r.student_number: r.status for r in records}
    
    # 4. 拼装结果
    result = []
    for stu in all_students:
        # 如果在记录里，就是对应状态；如果不在，就是“缺勤”
        status = record_map.get(stu.student_number, "absent") 
        result.append({
            "student_number": stu.student_number,
            "name": stu.name,
            "class_name": stu.class_name,
            "status": status
        })
    
    # 按学号排序
    result.sort(key=lambda x: x["student_number"])
    return result

# ==========================================
# 接口 7: 手动修改学生签到状态
# ==========================================
class StatusUpdate(BaseModel):
    student_number: str
    status: str  # present(已到), absent(缺勤), late(迟到), leave(请假)

@router.put("/sessions/{session_id}/records")
def update_attendance_status(
    session_id: int,
    update_data: StatusUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 查找是否已有记录
    record = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id,
        AttendanceRecord.student_number == update_data.student_number
    ).first()

    if record:
        # 如果有记录，直接改状态
        record.status = update_data.status
    else:
        # 如果没有记录（之前是缺勤），现在要补一条记录
        # 注意：这里需要 course_id 来验证学生是否存在，简化起见假设存在
        new_record = AttendanceRecord(
            session_id=session_id,
            student_number=update_data.student_number,
            status=update_data.status,
            device_info="teacher_manual" # 标记为老师手动修改
        )
        db.add(new_record)
    
    db.commit()
    return {"msg": "状态已更新"}

# ==========================================
# 接口 8: 导出 Excel
# ==========================================
@router.get("/sessions/{session_id}/export")
def export_session_report(
    session_id: int,
    db: Session = Depends(get_db),
    # 注意：下载接口通常用 URL 参数带 token，这里假设前端用了 fetch blob 下载带了 header
    current_user = Depends(get_current_user) 
):
    # 复用上面的逻辑获取完整列表
    data = get_session_detail(session_id, db, current_user)
    
    # 转换状态码为中文
    status_map = {
        "present": "✅ 已到",
        "absent": "❌ 缺勤",
        "late": "⚠️ 迟到",
        "leave": "🤒 请假"
    }
    
    # 准备 DataFrame
    df_data = []
    for item in data:
        df_data.append({
            "学号": item["student_number"],
            "姓名": item["name"],
            "班级": item["class_name"],
            "状态": status_map.get(item["status"], item["status"])
        })
        
    df = pd.DataFrame(df_data)
    
    # 写入内存 Buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='签到表')
        
    output.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="attendance_{session_id}.xlsx"'
    }
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ==========================================
# 接口 9: 获取课程历史统计 (用于画图)
# ==========================================
# course/router.py

# ... (前面的导入)

# ==========================================
# 接口: 获取课程历史出勤率 (用于图表)
# ==========================================
@router.get("/{course_id}/stats_history")
def get_course_stats_history(
    course_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 查出该课程所有的 Session，按时间正序排列
    sessions = db.query(CheckinSession).filter(
        CheckinSession.course_id == course_id
    ).order_by(CheckinSession.id).all()
    
    dates = []
    rates = []
    
    # 2. 获取总人数 (作为分母)
    total_students = db.query(Roster).filter(Roster.course_id == course_id).count()
    if total_students == 0:
        total_students = 1 # 防止除以0报错
        
    for s in sessions:
        # 【核心修复】如果 start_time 是 None (说明是预约状态或未开始)，直接跳过，不画在图上
        if not s.start_time:
            continue
            
        # 3. 统计实到人数
        present_count = db.query(AttendanceRecord).filter(
            AttendanceRecord.session_id == s.id,
            (AttendanceRecord.status == 'present') | (AttendanceRecord.status == 'late')
        ).count()
        
        # 4. 计算比例
        rate = round((present_count / total_students) * 100, 1)
        
        # 5. 格式化时间 (现在确认 s.start_time 不为空了，可以安全格式化)
        date_str = s.start_time.strftime("%m-%d %H:%M")
        
        dates.append(date_str)
        rates.append(rate)
        
    return {
        "dates": dates,
        "rates": rates
    }
# ==========================================
# 接口 10: 获取课程的历史签到列表 (含简要统计)
# ==========================================


@router.get("/{course_id}/sessions")
def get_course_sessions(
    course_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 按时间倒序查
    sessions = db.query(CheckinSession)\
        .filter(CheckinSession.course_id == course_id)\
        .order_by(desc(CheckinSession.id))\
        .all()
    
    result = []
    for s in sessions:
        # 1. 统计人数
        present_count = db.query(AttendanceRecord).filter(
            AttendanceRecord.session_id == s.id,
            AttendanceRecord.status != 'absent'
        ).count()
        
        # 2. 【核心优化】计算由后端定义的状态，前端只管渲染
        status = "unknown"
        display_time = ""
        
        if s.is_active:
            status = "running" # 进行中
            display_time = s.start_time.strftime("%m-%d %H:%M") if s.start_time else "进行中"
        elif s.start_time is None:
            status = "pending" # 预约了没开始
            display_time = s.scheduled_time.strftime("%m-%d %H:%M") + " (预约)"
        else:
            status = "finished" # 已结束
            display_time = s.start_time.strftime("%m-%d %H:%M")

        result.append({
            "id": s.id,
            "display_time": display_time, # 前端直接显示这个字符串
            "status": status,             # pending / running / finished
            "present_count": present_count
        })
        
    return result

@router.post("/sessions/{session_id}/stop")
def stop_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    session = db.query(CheckinSession).filter(CheckinSession.id == session_id).first()
    if not session: raise HTTPException(404, "Session not found")
    
    session.is_active = False
    session.end_time = datetime.now() # 记录结束时间
    db.commit()
    
    return {"msg": "签到已结束"}

# ===========================
# [新增] 接口: 手动重新触发资源生成
# ===========================
@router.post("/{course_id}/regenerate")
def regenerate_course_resources(
    request: Request,
    course_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    # 1. 权限校验
    course = db.query(Course).filter(
        Course.id == course_id, 
        Course.teacher_id == current_user.id
    ).first()
    
    if not course:
        raise HTTPException(status_code=404, detail="课程不存在")

    current_domain = str(request.base_url).rstrip("/")
    
    # 2. 检查是否正在运行 (防止重复点)
    # 如果内存里显示正在跑，就不让点
    if course_id in TASK_PROGRESS and TASK_PROGRESS[course_id]["status"] == "processing":
        raise HTTPException(status_code=400, detail="任务正在进行中，请勿重复提交")

    # 3. 强制重置进度状态 (为了让前端立马有反应)
    TASK_PROGRESS[course_id] = {
        "status": "processing",
        "progress": 0,
        "message": "正在重启任务..."
    }

    # 4. 重新加入后台队列
    background_tasks.add_task(generate_course_watermarks, course.id, current_domain)
    
    return {"msg": "任务已重启"}

# course/router.py

# ... (前面的导入)

# ==========================================
# 接口: 重置 Session 开始时间 (用于大屏加载完成后校准)
# ==========================================
@router.post("/sessions/{session_id}/reset_timer")
def reset_session_timer(
    session_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    session = db.query(CheckinSession).filter(CheckinSession.id == session_id).first()
    if not session:
        raise HTTPException(404, "Session not found")
    
    # 【核心】把开始时间更新为“现在”
    session.start_time = datetime.now()
    db.commit()
    
    return {"msg": "计时已重置", "new_start_time": session.start_time}