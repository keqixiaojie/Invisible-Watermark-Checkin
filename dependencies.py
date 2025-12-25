# dependencies.py
from database.models import engine
from sqlalchemy.orm import sessionmaker

# 初始化数据库连接会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 获取数据库会话 (Dependency)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# dependencies.py (追加内容)
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import SECRET_KEY, ALGORITHM
from database.models import User
# 这里的 get_db 是该文件上面已经定义好的，直接引用即可

# 定义 Token 获取方式：从请求头的 Authorization: Bearer <token> 获取
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# 【核心函数】获取当前登录用户
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 1. 解密 Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # 2. 去数据库查这个用户是否存在
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user