import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- 1. 학습된 파일 및 데이터 로드 ---
@st.cache_resource  # 모델과 스케일러를 매번 새로 읽어오지 않고 캐싱하여 성능을 높입니다.
def load_models_and_data():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('lung.csv')
    return model, scaler, df

try:
    model, scaler, df = load_models_and_data()
except Exception as e:
    st.error(f"파일을 읽어오는 중 오류가 발생했습니다. 파일이 같은 폴더에 있는지 확인해주세요: {e}")
    st.stop()

# --- 2. Streamlit UI 구성 ---
st.title("🏥 폐 건강 데이터 군집 예측 및 시각화")
st.write("환자의 정보를 입력하면 어떤 군집에 속하는지 예측하고 데이터 분포 상의 위치를 시각화합니다.")

st.sidebar.header("📋 환자 정보 입력")

# 새로운 환자 데이터 입력 받기
input_age = st.sidebar.number_input("나이 입력", min_value=0.0, max_value=120.0, value=40.0, step=1.0)
input_smoke = st.sidebar.slider("흡연량 입력", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
input_alkhol = st.sidebar.slider("음주량 입력", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

# 예측 및 시각화 버튼 생성
if st.sidebar.button("군집 예측하기"):
    
    st.subheader("🔮 예측 결과")
    
    # 입력 데이터를 DataFrame으로 변환
    new_patient = pd.DataFrame([[input_age, input_smoke, input_alkhol]], columns=['나이', '흡연량', '음주량'])
    
    # 스케일링 및 군집 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)
    
    # 결과 출력
    st.success(f"이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")
    
    st.markdown("---")
    
    # --- 3. 새로운 값의 위치 시각화 ---
    st.subheader("📊 데이터 분포 내 환자 위치")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 한글 폰트 깨짐 방지 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기준 (맥 사용 시 'AppleGothic'으로 변경)
    plt.rcParams['axes.unicode_minus'] = False

    # 기존 데이터 산점도 그리기 (lung.csv의 군집 컬럼명이 'c'라고 가정)
    if 'c' in df.columns:
        scatter = ax.scatter(df['나이'], df['흡연량'], c=df['c'], alpha=0.5, cmap='viridis', label='기존 환자 데이터')
        # 색상 바(Colorbar) 추가하여 군집 표시 도움
        fig.colorbar(scatter, ax=ax, label='군집 번호')
    else:
        # 혹시 'c' 컬럼이 없을 경우 기본 산점도 처리
        ax.scatter(df['나이'], df['흡연량'], alpha=0.3, color='gray', label='기존 환자 데이터')

    # 새 환자 위치 표시 (크기가 큰 검은색 X)
    ax.scatter(input_age, input_smoke, c='black', s=300, marker='X', label='입력된 환자')
    
    ax.set_xlabel('나이')
    ax.set_ylabel('흡연량')
    ax.set_title(f'나이 vs 흡연량 분포 (입력 환자 군집: {pred_cluster[0]}번)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 스트림릿 화면에 그래프 출력
    st.pyplot(fig)