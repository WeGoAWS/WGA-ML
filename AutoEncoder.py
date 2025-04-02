import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import joblib
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split  # 데이터셋 분할 추가


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 현재 사용 중인 디바이스: {device}")

# logs 불러오기
def load_logs_from_directory(directory):
    all_logs = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict) and "Records" in data:
                        all_logs.extend(data["Records"])
                    else:
                        if isinstance(data, list):
                            all_logs.extend(data)
                        else:
                            all_logs.append(data)
                except json.JSONDecodeError:
                    print(f"JSON decode error in file: {filename}")
    return all_logs

# AutoEncoder 모델 정의
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
            nn.Linear(128, input_dim)  # Sigmoid 제거
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 이상 탐지 함수 (공격 로그 탐지)
def detect_attack(model, encoder, scaler, new_log, threshold=0.1):
    expected_columns = ['userIdentity.type', 'eventName', 'eventSource', 'eventType', 'sourceIPAddress', 'userAgent']
    new_log['userIdentity.type'] = new_log['userIdentity']['type'] if isinstance(new_log.get('userIdentity'), dict) else 'Unknown'
    new_df = pd.DataFrame([{k: new_log.get(k, 'Unknown') for k in expected_columns}])
    encoded_new_data = encoder.transform(new_df[expected_columns])
    scaled_new_data = scaler.transform(encoded_new_data)
    X_new = torch.tensor(scaled_new_data, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        reconstructed_data = model(X_new)
        reconstruction_error = torch.mean(torch.abs(reconstructed_data - X_new), dim=1).cpu().numpy()
    # 임계값 이하이면 공격 로그로 판단하여 export
    if reconstruction_error <= threshold:
        return new_log
    else:
        return None

# 배치 학습 함수
model_global, encoder_global, scaler_global = None, None, None

def process_and_train_full(log_data, model=None, encoder=None, scaler=None):
    log_data = log_data[['userIdentity', 'eventType', 'eventSource']].copy()
    log_data.loc[:, 'userIdentity.type'] = log_data['userIdentity'].apply(lambda x: x.get('type') if isinstance(x, dict) else 'Unknown')
    log_data = log_data.drop(columns=['userIdentity'])

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(log_data[['userIdentity.type', 'eventType', 'eventSource']])
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(encoded_data)
    else:
        encoded_data = encoder.transform(log_data[['userIdentity.type', 'eventType', 'eventSource']])
        scaled_data = scaler.transform(encoded_data)

    X = torch.tensor(scaled_data, dtype=torch.float32)
    input_dim = X.shape[1]
    if model is None:
        model = AutoEncoder(input_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()  # Sigmoid 제거에 따라 손실함수 변경
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)

            if torch.isnan(batch).any() or torch.isinf(batch).any():
                print("배치에 NaN/Inf 있음!")
            loss = criterion(output, batch)
            if torch.isnan(loss):
                print("손실값이 NaN입니다!")

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(X)
        print(f"[Batch Training] Epoch {epoch+1}/5, Loss: {avg_loss:.6f}  (raw: {avg_loss})")

    return model, encoder, scaler

def process_and_train_in_batches(log_data, batch_size=5000):
    global model_global, encoder_global, scaler_global
    for start in range(0, len(log_data), batch_size):
        print(f"[Batch] 처리 중: {start} ~ {min(start+batch_size, len(log_data))}")
        batch = log_data.iloc[start:start+batch_size].copy()
        model_global, encoder_global, scaler_global = process_and_train_full(
            batch, model_global, encoder_global, scaler_global
        )
        torch.cuda.empty_cache()
        gc.collect()
    return model_global, encoder_global, scaler_global

# 시각화 함수 (UMAP 기반)
def visualize_latent_space(model, data_tensor, labels):
    model.eval()
    with torch.no_grad():
        latent = model.encoder(data_tensor.to(device)).cpu().numpy()

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    z = reducer.fit_transform(latent)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title('Latent Space Visualization (UMAP)')
    plt.colorbar(scatter, label='Label (0=Normal, 1=Anomaly)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.show()


# 메인 실행 로직
def main():
    print("[1] 로그 불러오는 중...")
    logs = load_logs_from_directory("./logs")
    df = pd.DataFrame(logs)
    print(f"총 로그 수: {len(df)}")

    print("[2] 전체 배치 학습 시작")
    model, encoder, scaler = process_and_train_in_batches(df)
    
    print("[3] 전처리 데이터 생성 중...")
    df = df[['userIdentity', 'eventType', 'eventSource']].copy()
    df['userIdentity.type'] = df['userIdentity'].apply(lambda x: x.get('type') if isinstance(x, dict) else 'Unknown')
    df = df.drop(columns=['userIdentity'])
    encoded_data = encoder.transform(df[['userIdentity.type', 'eventType', 'eventSource']])
    scaled_data = scaler.transform(encoded_data)
    X_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    labels = [1] * len(X_tensor)
    
    print("[4] 학습된 모델 저장 중...")
    torch.save(model.state_dict(), "./AutoEncoderModel/model.pth")
    joblib.dump(encoder, "./AutoEncoderModel/encoder.pkl")
    joblib.dump(scaler, "./AutoEncoderModel/scaler.pkl")
    print(" 모델 및 인코더, 스케일러 저장 완료")

if __name__ == "__main__":
    main()