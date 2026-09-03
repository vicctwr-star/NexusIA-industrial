import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="NEXUS",
    page_icon="",
    layout="wide"
)



st.title(" NEXUS - IA INDUSTRIAL")
st.subheader("Sistema Inteligente de Manutenção Preditiva")

st.divider()



from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)

n = 5000

temperatura = np.random.normal(60, 15, n)
vibracao = np.random.normal(0.6, 0.35, n)
pressao = np.random.normal(4.0, 0.5, n)
horas = np.random.uniform(100, 1000, n)
energia = np.random.normal(20, 5, n)

temperatura = np.clip(temperatura, 20, 110)
vibracao = np.clip(vibracao, 0, 2)
pressao = np.clip(pressao, 2, 5)
energia = np.clip(energia, 5, 40)



risco_continuo = (

    # Temperatura
    np.clip((temperatura - 40) / 50, 0, 1) * 0.25 +

    # Vibração
    np.clip((vibracao - 0.2) / 1.3, 0, 1) * 0.25 +

    # Pressão baixa
    np.clip((4.5 - pressao) / 1.5, 0, 1) * 0.20 +

    # Horas de uso
    np.clip((horas - 200) / 700, 0, 1) * 0.15 +

    # Consumo de energia
    np.clip((energia - 10) / 25, 0, 1) * 0.15
)

# Transformar em porcentagem
risco_percentual = risco_continuo * 100

# Classificação para treinamento
falha = (risco_percentual >= 70).astype(int)

df = pd.DataFrame({
    "temperatura": temperatura,
    "vibracao": vibracao,
    "pressao": pressao,
    "horas": horas,
    "energia": energia,
    "falha": falha
})



X = df[
    [
        "temperatura",
        "vibracao",
        "pressao",
        "horas",
        "energia"
    ]
]

y = df["falha"]




X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)




modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo.fit(
    X_treino,
    y_treino
)




previsoes = modelo.predict(X_teste)

precisao = accuracy_score(
    y_teste,
    previsoes
)


def calcular_risco(
    temperatura,
    vibracao,
    pressao,
    horas,
    energia
):

    risco = (

        np.clip(
            (temperatura - 40) / 50,
            0,
            1
        ) * 0.25 +

        np.clip(
            (vibracao - 0.2) / 1.3,
            0,
            1
        ) * 0.25 +

        np.clip(
            (4.5 - pressao) / 1.5,
            0,
            1
        ) * 0.20 +

        np.clip(
            (horas - 200) / 700,
            0,
            1
        ) * 0.15 +

        np.clip(
            (energia - 10) / 25,
            0,
            1
        ) * 0.15
    )

    return risco * 100

maquinas = pd.DataFrame({

    "Máquina": [
        "Máquina 01",
        "Máquina 02",
        "Máquina 03",
        "Máquina 04",
        "Máquina 05"
    ],

    "temperatura": [
        45,
        62,
        87,
        51,
        85
    ],

    "vibracao": [
        0.20,
        0.55,
        1.35,
        0.30,
        1.20
    ],

    "pressao": [
        4.2,
        4.0,
        3.2,
        4.3,
        3.3
    ],

    "horas": [
        250,
        450,
        910,
        300,
        500
    ],

    "energia": [
        12,
        19,
        31,
        14,
        20
    ]
})



dados = maquinas[
    [
        "temperatura",
        "vibracao",
        "pressao",
        "horas",
        "energia"
    ]
]

maquinas["risco"] = maquinas.apply(
    lambda x: calcular_risco(
        x["temperatura"],
        x["vibracao"],
        x["pressao"],
        x["horas"],
        x["energia"]
    ),
    axis=1
)



def status(risco):

    if risco < 30:
        return "🟢 NORMAL"

    elif risco < 70:
        return "🟡 ATENÇÃO"

    else:
        return "🔴 CRÍTICO"


maquinas["status"] = maquinas["risco"].apply(status)



col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Máquinas monitoradas",
        len(maquinas)
    )

with col2:
    criticas = sum(maquinas["risco"] >= 70)

    st.metric(
        "Máquinas críticas",
        criticas
    )

with col3:
    maior_risco = maquinas["risco"].max()

    st.metric(
        "Maior risco",
        f"{maior_risco:.1f}%"
    )

st.divider()

st.divider()

st.subheader("Desempenho da Inteligência Artificial")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dados utilizados",
        "5.000"
    )

with col2:
    st.metric(
        "Treinamento",
        "80% — 4.000"
    )

with col3:
    st.metric(
        "Teste",
        "20% — 1.000"
    )

st.write(
    f"### 🎯 Precisão do modelo: {precisao * 100:.2f}%"
)

st.caption(
    "O modelo foi treinado com 80% dos dados "
    "e avaliado com 20% de dados que não foram "
    "utilizados durante o treinamento."
)



st.subheader("🔧 Monitoramento das máquinas")

for _, maquina in maquinas.iterrows():

    st.write(
        f"**{maquina['Máquina']}** — "
        f"{maquina['status']} — "
        f"Risco: **{maquina['risco']:.1f}%**"
    )




st.divider()

st.subheader("🔎 Análise detalhada")

selecionada = st.selectbox(
    "Selecione uma máquina:",
    maquinas["Máquina"]
)

maquina_original = maquinas[
    maquinas["Máquina"] == selecionada
].iloc[0]



st.subheader("🎮 Cenários de operação")

st.write(
    "Escolha um cenário pré-configurado ou "
    "ajuste manualmente os sensores."
)




if "temperatura_manual" not in st.session_state:
    st.session_state.temperatura_manual = 45.0

if "vibracao_manual" not in st.session_state:
    st.session_state.vibracao_manual = 0.20

if "pressao_manual" not in st.session_state:
    st.session_state.pressao_manual = 4.2

if "horas_manual" not in st.session_state:
    st.session_state.horas_manual = 250

if "energia_manual" not in st.session_state:
    st.session_state.energia_manual = 12.0




col_normal, col_atencao, col_critico = st.columns(3)


with col_normal:

    if st.button(
        "🟢 NORMAL",
        use_container_width=True
    ):

        st.session_state.temperatura_manual = 45.0
        st.session_state.vibracao_manual = 0.20
        st.session_state.pressao_manual = 4.2
        st.session_state.horas_manual = 250
        st.session_state.energia_manual = 12.0


with col_atencao:

    if st.button(
        "🟡 ATENÇÃO",
        use_container_width=True
    ):

        st.session_state.temperatura_manual = 70.0
        st.session_state.vibracao_manual = 0.70
        st.session_state.pressao_manual = 3.7
        st.session_state.horas_manual = 650
        st.session_state.energia_manual = 24.0


with col_critico:

    if st.button(
        "🔴 CRÍTICO",
        use_container_width=True
    ):

        st.session_state.temperatura_manual = 90.0
        st.session_state.vibracao_manual = 1.45
        st.session_state.pressao_manual = 3.1
        st.session_state.horas_manual = 950
        st.session_state.energia_manual = 33.0




st.subheader("🎛️ Controle manual dos sensores")

st.write(
    "Você também pode alterar os valores individualmente "
    "para simular diferentes condições da máquina."
)


col1, col2 = st.columns(2)


with col1:

    st.slider(
        "🌡️ Temperatura (°C)",
        min_value=20.0,
        max_value=110.0,
        step=1.0,
        key="temperatura_manual"
    )

    st.slider(
        "📳 Vibração",
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        key="vibracao_manual"
    )

    st.slider(
        "💨 Pressão (bar)",
        min_value=2.0,
        max_value=5.0,
        step=0.1,
        key="pressao_manual"
    )


with col2:

    st.slider(
        "⏱️ Horas de uso",
        min_value=100,
        max_value=1000,
        step=10,
        key="horas_manual"
    )

    st.slider(
        "⚡ Consumo de energia",
        min_value=5.0,
        max_value=40.0,
        step=1.0,
        key="energia_manual"
    )




temperatura_atual = st.session_state.temperatura_manual
vibracao_atual = st.session_state.vibracao_manual
pressao_atual = st.session_state.pressao_manual
horas_atual = st.session_state.horas_manual
energia_atual = st.session_state.energia_manual




risco_atual = calcular_risco(
    temperatura_atual,
    vibracao_atual,
    pressao_atual,
    horas_atual,
    energia_atual
)




if risco_atual < 30:

    status_atual = "🟢 NORMAL"

elif risco_atual < 70:

    status_atual = "🟡 ATENÇÃO"

else:

    status_atual = "🔴 CRÍTICO"




st.divider()

st.subheader("📊 Dados atuais da máquina")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "🌡️ Temperatura",
        f"{temperatura_atual:.0f} °C"
    )


with col2:

    st.metric(
        "📳 Vibração",
        f"{vibracao_atual:.2f}"
    )


with col3:

    st.metric(
        "💨 Pressão",
        f"{pressao_atual:.1f} bar"
    )


with col4:

    st.metric(
        "⏱️ Horas",
        f"{horas_atual:.0f} h"
    )


with col5:

    st.metric(
        "⚡ Energia",
        f"{energia_atual:.0f} kWh"
    )




st.divider()

st.subheader("⚠️ Análise de risco")


st.write(
    f"## {status_atual}"
)


st.progress(
    int(risco_atual)
)


st.write(
    f"### Risco de falha: {risco_atual:.1f}%"
)




if risco_atual >= 70:

    st.error(
        "🔴 RISCO CRÍTICO — "
        "Realizar manutenção preventiva imediatamente."
    )

elif risco_atual >= 30:

    st.warning(
        "🟡 ATENÇÃO — "
        "Recomenda-se realizar uma inspeção preventiva."
    )

else:

    st.success(
        "🟢 NORMAL — "
        "Máquina operando dentro dos padrões."
    )




