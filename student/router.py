# student/router.py
import base64
import cv2
import numpy as np
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database.models import CheckinSession, AttendanceRecord, Roster
from dependencies import get_db
from config import SECRET_KEY, ALGORITHM
from jose import jwt, JWTError

# 引入你的验证服务
from utils.verification_service import SignInVerificationService
from starlette.concurrency import run_in_threadpool

router = APIRouter(
    prefix="/student",
    tags=["学生端 (Student)"]
)

# 初始化验证器 (单例)
verifier = SignInVerificationService()

# --- 数据模型 ---
class VerifyRequest(BaseModel):
    course_id: int
    frames: List[str]  # Base64 字符串列表
    client_timestamp: float = 0.0 # 新增字段

class CheckinSubmit(BaseModel):
    verify_token: str
    student_number: str
    name: str

# ===========================
# 接口 1: 验证图片序列 (水印验证)
# ===========================
@router.post("/verify")
async def verify_scan(req: VerifyRequest, db: Session = Depends(get_db)):
    # 1. 检查活跃会话
    session = db.query(CheckinSession).filter(
        CheckinSession.course_id == req.course_id,
        CheckinSession.is_active == True
    ).first()
    
    if not session:
        # 调试用：如果没有活跃会话，手动造一个假开始时间测试
        # t_start = time.time() - 30 
        raise HTTPException(status_code=400, detail="当前没有正在进行的签到")
    else:
        t_start = session.start_time.timestamp()

    # 2. 确定提交时间 (解决 8s 误差的核心)
    # 如果前端传了拍照时间，就用前端的；否则用服务器收到请求的时间
    if req.client_timestamp and req.client_timestamp > 0:
        t_submit = req.client_timestamp
    else:
        t_submit = time.time()

    # 3. Base64 解码 (保持不变)
    frames_np = []
    try:
        for b64 in req.frames:
            if ',' in b64: b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None: frames_np.append(img)
    except:
        raise HTTPException(400, "图片格式错误")

    if not frames_np:
        raise HTTPException(400, "未接收到有效图片")

    # 4. 调用验证服务 (并行计算交给 Service 内部处理)
    # 这里使用 run_in_threadpool 是为了把这一整块逻辑扔出主线程，
    # 虽然 Service 内部也开了线程池，但这一层调度本身也是耗时的。
    try:
        report = await run_in_threadpool(
            verifier.verify_sign_in,
            user_id="anonymous",
            course_id=req.course_id,
            frames=frames_np,
            start_time=t_start,
            submit_time=t_submit  # <--- 这里传入了对齐后的时间
        )
    except Exception as e:
        print(f"Verify Error: {e}")
        raise HTTPException(500, "验证服务内部错误")

    if not report["success"]:
        return {"success": False, "reason": report["reason"]}

    # 5. 签发 Token (保持不变)
    payload = {
        "sub": "verified_client",
        "sid": session.id,
        "cid": req.course_id,
        "exp": datetime.utcnow() + timedelta(minutes=5)
    }
    verify_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return {"success": True, "verify_token": verify_token}
# ... (checkin 接口保持不变) ...
  

# ===========================
# 接口 2: 提交个人信息 (最终入库)
# ===========================
from datetime import datetime, timedelta

@router.post("/checkin")
def submit_checkin(
    data: CheckinSubmit,
    db: Session = Depends(get_db)
):
    # 1. 校验 Verify Token
    try:
        payload = jwt.decode(data.verify_token, SECRET_KEY, algorithms=[ALGORITHM])
        session_id = payload.get("sid")
    except JWTError:
        raise HTTPException(status_code=401, detail="验证凭证已失效，请重新扫码")

    # 2. 再次检查 Session 状态 (防止扫码后过了很久才提交)
    session = db.query(CheckinSession).filter(CheckinSession.id == session_id).first()
    if not session or not session.is_active:
        raise HTTPException(status_code=400, detail="签到已结束")

    # 3. 检查学生是否在花名册中 (防呆)
    roster = db.query(Roster).filter(
        Roster.course_id == session.course_id,
        Roster.student_number == data.student_number
    ).first()

    if not roster:
        raise HTTPException(status_code=400, detail="非本课程学生，请检查学号")
    
    # 可选：简单校验姓名
    if roster.name != data.name:
        raise HTTPException(status_code=400, detail=f"学号 {data.student_number} 对应的姓名不是 {data.name}")

    # 4. 检查是否重复签到
    exists = db.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id,
        AttendanceRecord.student_number == data.student_number
    ).first()

    if exists:
        return {"success": True, "msg": "您已经签到过了，无需重复提交"}

    # 5. 写入数据库
    new_record = AttendanceRecord(
        session_id=session_id,
        student_number=data.student_number,
        status="present",
        device_info="mobile_web"
    )
    db.add(new_record)
    db.commit()

    return {"success": True, "msg": "签到成功！"}