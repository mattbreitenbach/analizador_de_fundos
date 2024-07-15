
from datetime import datetime
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analise_risco as ar


@st.cache_data 
def carregar_cotas_fundos(nome_pasta: str, nome_arquivo: str) -> pd.DataFrame:
    df = pd.read_excel(f"{nome_pasta}/{nome_arquivo}.xlsx", parse_dates=["DATA"])
    df.set_index("DATA", inplace=True)
    return df


st.set_page_config(
    page_title="Analizador de Fundos",
)

st.title("Analizador de Fundos", anchor=False)
cotas = carregar_cotas_fundos("./assets", "cotas_fundos")

#Selecionar os fundos para a análise
lista_de_fundos = cotas.columns.to_list()
fundos_multiselect = st.multiselect("Selecione até 7 fundos", tuple(lista_de_fundos), placeholder="Fundos")
if fundos_multiselect:
    if len(fundos_multiselect) > 7:
        st.text("Selecione até 7 fundos.")
        cotas = cotas[random.sample(lista_de_fundos, 5)]
    else:
        cotas = cotas[fundos_multiselect]
else:
    cotas = cotas[random.sample(lista_de_fundos, 5)]


#Selecionar a página desejada
paginas_disponiveis = ["Retorno", "Retorno X Volatilidade", "Drawdown", "VaR"]
pagina_select = st.selectbox("Selecionar Página", tuple(paginas_disponiveis), label_visibility="collapsed", placeholder="Páginas")


if pagina_select == "Retorno":
    cotas_normalizadas = cotas.divide(cotas.iloc[0]).subtract(1).multiply(100)
    cotas_normalizadas = cotas_normalizadas.round(2)
    fig = go.Figure()
    for coluna in cotas_normalizadas.columns:
        fig.add_trace(go.Scatter(x=cotas_normalizadas.index, y=cotas_normalizadas[coluna], mode="lines", name=coluna))
    
    fig.update_layout(
        title="Retorno dos Fundos",
        xaxis_title="Data",
        yaxis_title="Retornos",
        hovermode='x',
        legend=dict(
            orientation='h',
            yanchor='bottom',  
            y=1.02,
            xanchor='right', 
            x=1)
    )
    st.plotly_chart(fig)


if pagina_select == "Retorno X Volatilidade":
    retornos = cotas.iloc[-1]/cotas.iloc[0]
    cagr = ((retornos**(len(retornos)/252))-1).multiply(100)
    cagr = cagr.round(2)
    cagr.sort_values(ascending=False, inplace=True)
    volatilidades = cotas.apply(ar.calcular_vol).multiply(100)
    volatilidades = volatilidades.round(2)
    df_ret_vol = cagr.to_frame(name="CAGR")
    df_ret_vol["Volatilidade"] = volatilidades
    df_ret_vol["Fundo"] = df_ret_vol.index

    fig = px.scatter(df_ret_vol, x="Volatilidade", y="CAGR", text="Fundo", title="Retorno e Volatilidade")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig)    

    st.dataframe(df_ret_vol.style.format({"CAGR": "{:,.2f}%", "Volatilidade": "{:,.2f}%"}),
                width=800,
                height=250,
                column_order=("Fundo", "CAGR", "Volatilidade"),
                hide_index=True)
    

if pagina_select == "Drawdown":
    drawdown = cotas.apply(ar.calcular_drawdown_hist).multiply(100)
    drawdown = drawdown.round(2)

    fig = go.Figure()
    for coluna in drawdown.columns:
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown[coluna], mode="lines", name=coluna))
    fig.update_layout(
        title="Drawdown dos Fundos",
        xaxis_title="Data",
        yaxis_title="Drawdown",
        hovermode='x',
        legend=dict(
            orientation='h',
            yanchor='bottom',  
            y=1.02,
            xanchor='right', 
            x=1)
    )
    st.plotly_chart(fig)


if pagina_select == "VaR":
    ic = st.slider("Qual é o Intervalo de Confiança?", min_value=90.0, max_value=100.0, value = 95.0)

    for coluna in cotas.columns:
        fundo_retorno_serie = cotas[coluna].pct_change().dropna()*100
        var_hist = ar.calcular_var_hist(cotas[coluna], alpha=ic)*100
        var_param = ar.calcular_var_param(cotas[coluna], alpha=ic)*100
        fig = px.histogram(fundo_retorno_serie, x=coluna, nbins=200, histnorm='probability density', title=coluna)

        fig.add_trace(go.Scatter(x=[var_hist, var_hist], y=[0, 0.3], mode='lines', name='VaR histórico', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=[var_param, var_param], y=[0, 0.3], mode='lines', name='VaR Paramétrico', line=dict(color='blue', width=2)))

        fig.update_layout(
            xaxis_title='Retorno',
            yaxis_title='Densidade',
            showlegend=True
        )
        st.plotly_chart(fig)
