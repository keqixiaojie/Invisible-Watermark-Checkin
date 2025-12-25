import os
from pathlib import Path

# 1. 设置基准路径为当前运行目录 (相对路径模式)
# 注意：这要求你必须在项目根目录下运行 main.py (即 python main.py)
BASE_DIR = Path(".")

# 2. 数据库文件夹: ./database/datas
DB_DIR = BASE_DIR / "database" / "datas"

# 3. 模型文件夹 (相对路径)
# 最终路径会变成: utils
DeepModel_FOLDER = BASE_DIR / "utils" / "model"

# 4. QR模型文件夹
QRMOAEL_FOLDER = BASE_DIR / "QR_watermark" / "QR_model"

# 自动创建数据库目录
if not DB_DIR.exists():
    DB_DIR.mkdir(parents=True, exist_ok=True)

# 数据库文件路径: ./database/datas/attendance.db
DB_FILE_PATH = DB_DIR / "attendance.db"

# SQLAlchemy 连接字符串
# sqlite:///database/datas/attendance.db (注意这里是3个斜杠，表示相对路径)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE_PATH.as_posix()}"

# JWT 配置
SECRET_KEY = "my_super_secret_key_change_this_in_production"
ALGORITHM = "HS256"
# 开发阶段建议改长一点，免得频繁登录
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7天

if __name__ == "__main__":
    print(f"项目根目录(相对): {BASE_DIR}")
    print(f"数据库路径: {DB_FILE_PATH}")
    print(f"模型路径示例: {DeepModel_FOLDER / 'digit_recognizer.onnx'}")