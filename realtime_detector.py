import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 현재 사용 중인 디바이스: {device}")

# AutoEncoder 정의 (모델 불러오기용)
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 이상 탐지 함수 (공격 로그 유사 판별)
max_error = 0  # 전역 최대 오차 변수 추가
def detect_attack(model, encoder, scaler, new_log, threshold=0.03):
    global max_error
    expected_columns = ['userIdentity.type', 'eventType', 'eventSource']

    identity = new_log.get('userIdentity', {})
    new_log['userIdentity.type'] = identity.get('type', 'Unknown') if isinstance(identity, dict) else 'Unknown'

    new_df = pd.DataFrame([{k: new_log.get(k, 'Unknown') for k in expected_columns}])
    encoded_new_data = encoder.transform(new_df[expected_columns])
    scaled_new_data = scaler.transform(encoded_new_data)
    X_new = torch.tensor(scaled_new_data, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(X_new)
        reconstructed_data = torch.sigmoid(output)
        reconstruction_error = torch.mean(torch.abs(reconstructed_data - X_new), dim=1).cpu().numpy()[0]

    if reconstruction_error > max_error:
        max_error = reconstruction_error

    print(f"🔍 재구성 오차: {reconstruction_error:.6f} (현재 최고: {max_error:.6f})")

    if reconstruction_error <= threshold:
        print(" 공격 로그로 판단되어 export 예정")
        return new_log
    else:
        print(" 정상 로그로 판단되어 제외")
        return None
    
# 실시간 로그 감지 및 공격 로그 저장
class MyHandler(FileSystemEventHandler):
    def __init__(self, model, encoder, scaler, export_path="attack_logs.json"):
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.export_path = export_path
        self.attack_logs = []

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            print(f"New log file detected: {event.src_path}")
            time.sleep(10)
            try:
                with open(event.src_path, 'r') as f:
                    log = json.load(f)
                    if isinstance(log, dict) and "Records" in log:
                        records = log["Records"]
                    elif isinstance(log, list):
                        records = log
                    else:
                        records = [log]
                    for entry in records:
                        identity = entry.get('userIdentity', {})
                        entry['userIdentity.type'] = identity.get('type', 'Unknown') if isinstance(identity, dict) else 'Unknown'

                        attack = detect_attack(self.model, self.encoder, self.scaler, entry)
                        if attack:
                            self.attack_logs.append(attack)

                    if self.attack_logs:
                        if os.path.exists(self.export_path):
                            with open(self.export_path, 'r') as prev:
                                try:
                                    previous_logs = json.load(prev)
                                except:
                                    previous_logs = []
                            self.attack_logs = previous_logs + self.attack_logs

                        with open(self.export_path, "w") as f:
                            json.dump(self.attack_logs, f, indent=4)

                        print(f" 누적된 공격 로그 {len(self.attack_logs)}건 저장됨 → {self.export_path}")
            except Exception as e:
                print(f"오류 발생: {e}")
                
def main():
    print("모델 로드 중...")
    encoder = joblib.load("./AutoEncoderModel/encoder.pkl")
    scaler = joblib.load("./AutoEncoderModel/scaler.pkl")
    dummy_input = torch.zeros((1, encoder.transform([["Unknown", "Unknown", "Unknown"]]).shape[1]))
    model = AutoEncoder(dummy_input.shape[1]).to(device)
    model.load_state_dict(torch.load("./AutoEncoderModel/model.pth", map_location=device))
    print(" 모델 및 전처리기 불러오기 완료")
    
    print("로그 감지 시작")
    event_handler = MyHandler(model, encoder, scaler)
    observer = Observer()
    observer.schedule(event_handler, path="./flaws_cloudtrail_logs", recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()