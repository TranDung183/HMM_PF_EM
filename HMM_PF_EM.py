import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from filterpy.monte_carlo import systematic_resample
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === BƯỚC 1: Đọc dữ liệu ===
df = pd.read_excel("C:/CODE/yahoo_data.xlsx")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# === BƯỚC 2: Chọn đặc trưng và chuẩn hóa ===
features = df[['Open', 'High', 'Low', 'Close']].dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Cột Close riêng cho Particle Filter
close_prices = df['Close'].dropna().values.reshape(-1, 1)
scaler_close = StandardScaler()
close_scaled = scaler_close.fit_transform(close_prices).flatten()

# === BƯỚC 3: Huấn luyện HMM với EM ===
model_em = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
model_em.fit(features_scaled)
hidden_states_em = model_em.predict(features_scaled)

# Đánh giá HMM bằng Log-Likelihood
log_likelihood = model_em.score(features_scaled)
print(f"Log-Likelihood của HMM: {log_likelihood:.4f}")

# === BƯỚC 4: Huấn luyện bằng Particle Filter (trên Close) ===
def transition(x):
    return x + np.random.normal(0, 0.2)

def particle_filter(observations, N=1000):
    T = len(observations)
    particles = np.random.randn(N)
    weights = np.ones(N) / N
    estimates = []

    for t in range(T):
        particles = transition(particles)
        weights *= np.exp(-0.5 * ((observations[t] - particles) ** 2))
        weights += 1e-300
        weights /= np.sum(weights)

        estimate = np.sum(particles * weights)
        estimates.append(estimate)

        indices = systematic_resample(weights)
        particles = particles[indices]
        weights = np.ones(N) / N

    return np.array(estimates)

estimates_pf = particle_filter(close_scaled, N=1000)

# === BƯỚC 5: Đánh giá Particle Filter ===
mae_pf = mean_absolute_error(close_scaled, estimates_pf)
rmse_pf = np.sqrt(mean_squared_error(close_scaled, estimates_pf))
std_pf = np.std(estimates_pf)

print(f"MAE của Particle Filter: {mae_pf:.4f}")
print(f"RMSE của Particle Filter: {rmse_pf:.4f}")
print(f"Độ lệch chuẩn của Particle Filter: {std_pf:.4f}")

# === BƯỚC 6: Trực quan hóa so sánh PF và HMM ===
close_scaled_hmm = features_scaled[:, 3]  # Cột Close sau chuẩn hóa

fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# --- Subplot 1: Particle Filter ---
axs[0].plot(close_scaled, label="Close thực tế", alpha=0.6)
axs[0].plot(estimates_pf, label="Dự báo Particle Filter", alpha=0.8)
axs[0].set_title("Dự báo bằng Particle Filter")
axs[0].set_xlabel("Chỉ mục thời gian")
axs[0].set_ylabel("Close (chuẩn hóa)")
axs[0].legend()
axs[0].grid(True)

# --- Subplot 2: HMM Hidden States ---
for i in range(model_em.n_components):
    idx = (hidden_states_em == i)
    axs[1].plot(np.arange(len(close_scaled_hmm))[idx], close_scaled_hmm[idx], '.', label=f"State {i}")
axs[1].set_title("Hidden States từ HMM (EM) trên biến Close")
axs[1].set_xlabel("Chỉ mục thời gian")
axs[1].set_ylabel("Close (chuẩn hóa)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
