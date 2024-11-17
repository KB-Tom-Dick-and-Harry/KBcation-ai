from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# 데이터베이스 구성 정보
db_config = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

answer_examples=[
    {
        "input" : "너의 이름은 뭐야?",
        "answer" : "KB 챗봇입니다."
    }
]