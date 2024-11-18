import subprocess
import sys
from threading import Thread

def run_server(command):
    subprocess.run(command, shell=True)

def main():
    servers = [
        "uvicorn llm_server:app --host 0.0.0.0 --port 8000",
        "uvicorn game_server:app --host 0.0.0.0 --port 8001"
    ]
    
    threads = [Thread(target=run_server, args=(cmd,)) for cmd in servers]
    
    for t in threads:
        t.start()
    
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n서버 종료 중...")
        sys.exit(0)

if __name__ == "__main__":
    main()