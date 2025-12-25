from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# [新增 1] 导入 CORS 中间件
from fastapi.middleware.cors import CORSMiddleware 

from auth import router as auth_router
from course import router as course_router
from student import router as student_router

app = FastAPI(title="签到系统后端")

# [新增 2] 配置 CORS (必须加在 app = FastAPI() 之后，路由挂载之前)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" 代表允许任何网址访问 (localhost, ngrok, 手机IP等)
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法 (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # 允许所有 Header
)

# 挂载 API 路由
app.include_router(auth_router.router)
app.include_router(course_router.router)
app.include_router(student_router.router)

# 挂载静态文件
app.mount("/pages", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def root():
    return {"message": "Server is running!"}