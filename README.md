# Invisible-Watermark-Checkin
BNU数字图像处理大作业，目标是实现一个利用隐形水印技术实现现场签到的小程序

### 运行方式
1. 首先确保工作目录为项目根目录
2. 安装环境
    - pip install opencv-contrib-python
    - pip install sqlalchemy
    - pip install fastapi uvicorn[standard] python-jose[cryptography] passlib[bcrypt] python-multipart
    - ……
3. 运行命令：uvicorn main:app --reload
4. 在运行的域名后加/pages/teacher_dashboard.html