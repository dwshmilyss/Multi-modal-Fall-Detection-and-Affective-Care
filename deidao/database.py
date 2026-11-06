# database.py
'''
数据库初始化模块核心功能：
负责 MySQL 数据库的自动连接与基础表结构的初始化。
主要内容：自动创建 fall_detector_db 数据库。  
创建 users（用户权限表）和 alarm_logs（跌倒告警日志表）。  
默认写入 root（管理员）和 family（家属）的初始账号密码数据。
'''
import os
import pymysql
from pathlib import Path

# 加载 .env 文件（如果存在）
def _load_dotenv():
    """从项目根目录加载 .env 文件到环境变量"""
    try:
        # 查找项目根目录的 .env 文件
        env_paths = [
            Path(__file__).parent.parent / '.env',  # 项目根目录
            Path(__file__).parent / '.env',         # deidao 目录
        ]
        for env_path in env_paths:
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # 只在环境变量不存在时设置（环境变量优先）
                            if key and key not in os.environ:
                                os.environ[key] = value
                break
    except Exception:
        pass

# 模块加载时自动加载 .env
_load_dotenv()

# 从环境变量读取数据库配置，提供默认值
DB_HOST = os.environ.get('MYSQL_HOST', 'localhost')
DB_PORT = int(os.environ.get('MYSQL_PORT', '3306'))
DB_USER = os.environ.get('MYSQL_USER', 'root')
DB_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
DB_NAME = os.environ.get('MYSQL_DATABASE', 'fall_detector_db')

def init_database():
    """自动初始化 MySQL 数据库与基础表结构"""
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} DEFAULT CHARACTER SET utf8mb4;")
        cursor.close()
        conn.close()

        db = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4'
        )
        cursor = db.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(100) NOT NULL,
            role VARCHAR(20) NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alarm_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            alarm_time VARCHAR(50),
            video_filename VARCHAR(255),
            status VARCHAR(50)
        );
        """)

        cursor.execute("SELECT COUNT(*) FROM users WHERE username='root';")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO users (username, password, role) VALUES ('root', '123456', 'admin');")
            cursor.execute("INSERT INTO users (username, password, role) VALUES ('family', '123456', 'family');")
            db.commit()

        cursor.close()
        db.close()
        print(" MySQL 数据库连接及初始化成功！")
        return True
    except Exception as e:
        print(f" 数据库连接失败: {e}")
        return False