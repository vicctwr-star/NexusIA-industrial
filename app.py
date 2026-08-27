import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# -----------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------

st.set_page_config(
    page_title="NEXUS",
    page_icon="",
    layout="wide"
)

# -----------------------------------
# TÍTULO
# -----------------------------------

st.title(" NEXUS - IA INDUSTRIAL")
st.subheader("Sistema Inteligente de Manutenção Preditiva")

st.divider()

# -----------------------------------
# CRIANDO MODELO SIMPLES
# -----------------------------------

from sklearn.ensemble import RandomForestClassifier

# Dataset artificial
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

risco = (
    (temperatura > 75).astype(int) +
    (vibracao > 0.9).astype(int) +
    (pressao < 3.6).astype(int) +
    (horas > 750).astype(int) +
    (energia > 27).astype(int)
)

falha = (risco >= 3).astype(int)

df = pd.DataFrame({
    "temperatura": temperatura,
    "vibracao": vibracao,
    "pressao": pressao,
    "horas": horas,
    "energia": energia,
    "falha": falha
})

# -----------------------------------
# TREINAMENTO E TESTE
# -----------------------------------

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


# Separar os dados:
# 80% para treinamento
# 20% para teste

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# TREINAR MODELO
# -----------------------------------

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo.fit(
    X_treino,
    y_treino
)


# -----------------------------------
# TESTAR MODELO
# -----------------------------------

previsoes = modelo.predict(X_teste)

precisao = accuracy_score(
    y_teste,
    previsoes
)
# -----------------------------------
# MÁQUINAS DA FÁBRICA
# -----------------------------------

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
        76
    ],

    "vibracao": [
        0.20,
        0.55,
        1.35,
        0.30,
        0.95
    ],

    "pressao": [
        4.2,
        4.0,
        3.2,
        4.3,
        3.5
    ],

    "horas": [
        250,
        450,
        910,
        300,
        780
    ],

    "energia": [
        12,
        19,
        31,
        14,
        28
    ]
})

# -----------------------------------
# PREVISÃO
# -----------------------------------

dados = maquinas[
    [
        "temperatura",
        "vibracao",
        "pressao",
        "horas",
        "energia"
    ]
]

maquinas["risco"] = (
    modelo.predict_proba(dados)[:, 1] * 100
)

# -----------------------------------
# STATUS
# -----------------------------------

def status(risco):

    if risco < 30:
        return "🟢 NORMAL"

    elif risco < 70:
        return "🟡 ATENÇÃO"

    else:
        return "🔴 CRÍTICO"


maquinas["status"] = maquinas["risco"].apply(status)

# -----------------------------------
# INDICADORES
# -----------------------------------

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

# -----------------------------------
# STATUS DAS MÁQUINAS
# -----------------------------------

st.subheader("🔧 Monitoramento das máquinas")

for _, maquina in maquinas.iterrows():

    st.write(
        f"**{maquina['Máquina']}** — "
        f"{maquina['status']} — "
        f"Risco: **{maquina['risco']:.1f}%**"
    )

# -----------------------------------
# SELECIONAR MÁQUINA
# -----------------------------------

st.divider()

st.subheader("🔎 Análise detalhada")

selecionada = st.selectbox(
    "Selecione uma máquina:",
    maquinas["Máquina"]
)

# -----------------------------------
# CONTROLES DE SIMULAÇÃO
# -----------------------------------

if "falha_simulada" not in st.session_state:
    st.session_state.falha_simulada = False

col_botao1, col_botao2 = st.columns(2)

with col_botao1:

    if st.button("🚨 SIMULAR FALHA", use_container_width=True):
        st.session_state.falha_simulada = True

with col_botao2:

    if st.button("🟢 VOLTAR AO NORMAL", use_container_width=True):
        st.session_state.falha_simulada = False


# -----------------------------------
# PEGAR DADOS DA MÁQUINA
# -----------------------------------

maquina_original = maquinas[
    maquinas["Máquina"] == selecionada
].iloc[0]


# -----------------------------------
# APLICAR SIMULAÇÃO
# -----------------------------------

if st.session_state.falha_simulada:

    maquina = maquina_original.copy()

    maquina["temperatura"] = 90
    maquina["vibracao"] = 1.45
    maquina["pressao"] = 3.1
    maquina["horas"] = 950
    maquina["energia"] = 33

    # Recalcular o risco usando o modelo
    dados_simulados = pd.DataFrame({
        "temperatura": [maquina["temperatura"]],
        "vibracao": [maquina["vibracao"]],
        "pressao": [maquina["pressao"]],
        "horas": [maquina["horas"]],
        "energia": [maquina["energia"]]
    })

    maquina["risco"] = (
        modelo.predict_proba(dados_simulados)[0][1] * 100
    )

    maquina["status"] = status(maquina["risco"])

else:

    maquina = maquina_original.copy()
# -----------------------------------
# DADOS DA MÁQUINA
# -----------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🌡️ Temperatura",
        f"{maquina['temperatura']} °C"
    )

with col2:
    st.metric(
        "📳 Vibração",
        f"{maquina['vibracao']}"
    )

with col3:
    st.metric(
        "💨 Pressão",
        f"{maquina['pressao']} bar"
    )

with col4:
    st.metric(
        "⏱️ Horas",
        f"{maquina['horas']:.0f} h"
    )

with col5:
    st.metric(
        "⚡ Energia",
        f"{maquina['energia']} kWh"
    )

# -----------------------------------
# RISCO
# -----------------------------------

st.subheader("⚠️ Análise de risco")

risco_atual = maquina["risco"]

st.progress(
    int(risco_atual)
)

st.write(
    f"### Risco de falha: {risco_atual:.1f}%"
)

# -----------------------------------
# RECOMENDAÇÃO
# -----------------------------------

if risco_atual >= 70:

    st.error(
        "🔴 RISCO CRÍTICO — "
        "Realizar manutenção preventiva."
    )

elif risco_atual >= 30:

    st.warning(
        "🟡 ATENÇÃO — "
        "Recomenda-se realizar uma inspeção."
    )

else:

    st.success(
        "🟢 NORMAL — "
        "Máquina operando dentro dos padrões."
    )