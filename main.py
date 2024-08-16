import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

import analise_risco as ar


@st.cache_data
def carregar_cotas_fundos(nome_pasta: str, nome_arquivo: str) -> pd.DataFrame:
    df = pd.read_excel(
        f"{nome_pasta}/{nome_arquivo}.xlsx", parse_dates=["DATA"])
    df.set_index("DATA", inplace=True)
    return df


st.set_page_config(
    page_title="Analizador de Fundos",
)

st.title("Analizador de Fundos", anchor=False)
cotas = carregar_cotas_fundos("./assets", "cotas_fundos")

# Selecionar os fundos para a análise
lista_de_fundos = cotas.columns.to_list()
fundos_multiselect = st.multiselect("Selecione até 7 fundos",
                                    tuple(lista_de_fundos),
                                    placeholder="Fundos")
if fundos_multiselect:
    if len(fundos_multiselect) > 7:
        st.text("Selecione até 7 fundos.")
        cotas = cotas[random.sample(lista_de_fundos, 7)]
    else:
        cotas = cotas[fundos_multiselect]
else:
    cotas = cotas[random.sample(lista_de_fundos, 5)]

retornos = cotas.pct_change().dropna()
log_retornos = np.log(retornos)

# Selecionar a página desejada
paginas_disponiveis = ["Retorno",
                       "Correlação",
                       "Retorno X Volatilidade",
                       "Drawdown",
                       "VaR",
                       "Markowitz"]
pagina_select = st.selectbox("Selecionar Página",
                             tuple(paginas_disponiveis),
                             label_visibility="collapsed",
                             placeholder="Páginas")


if pagina_select == "Retorno":
    cotas_normalizadas = cotas.divide(cotas.iloc[0]).subtract(1).multiply(100)
    cotas_normalizadas = cotas_normalizadas.round(2)
    fig = go.Figure()

    for coluna in cotas_normalizadas.columns:
        fig.add_trace(go.Scatter(x=cotas_normalizadas.index,
                                 y=cotas_normalizadas[coluna],
                                 mode="lines",
                                 name=coluna))

    fig.update_layout(title="Retorno dos Fundos",
                      xaxis_title="Data",
                      yaxis_title="Retornos",
                      hovermode='x',
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  y=1.02,
                                  xanchor='right',
                                  x=1))

    st.plotly_chart(fig)

if pagina_select == "Correlação":
    retornos.columns = [col[:15] for col in retornos.columns]
    matriz_corr = retornos.corr()

    corr_fig = px.imshow(matriz_corr,
                         color_continuous_scale='Blues',
                         text_auto=True)
    corr_fig.update_layout(width=800,
                           height=800,
                           xaxis=dict(tickangle=-70))
    st.plotly_chart(corr_fig)

    dendro_fig = ff.create_dendrogram(retornos.T,
                                      color_threshold=0.5,
                                      labels=retornos.columns.tolist())
    dendro_fig.update_traces(line_width=5)
    dendro_fig.update_layout(width=800,
                             height=800,
                             title="Dendrograma de Correlação entre Fundos",
                             yaxis=dict(title='',
                                        showticklabels=False),
                             xaxis=dict(tickangle=-70))
    st.plotly_chart(dendro_fig)

if pagina_select == "Retorno X Volatilidade":
    retornos_no_periodo = cotas.iloc[-1]/cotas.iloc[0]
    cagr = ((retornos_no_periodo**(len(retornos_no_periodo)/252))-1).multiply(100)
    cagr = cagr.round(2)
    cagr.sort_values(ascending=False, inplace=True)
    volatilidades = retornos.apply(ar.calcular_vol).multiply(100)
    volatilidades = volatilidades.round(2)
    df_ret_vol = cagr.to_frame(name="CAGR")
    df_ret_vol["Volatilidade"] = volatilidades
    df_ret_vol["Fundo"] = df_ret_vol.index

    fig_retxvol = px.scatter(df_ret_vol,
                             x="Volatilidade",
                             y="CAGR",
                             text="Fundo",
                             title="Retorno e Volatilidade")
    fig_retxvol.update_traces(textposition="top center")
    st.plotly_chart(fig_retxvol)

    st.dataframe(df_ret_vol.style.format({"CAGR": "{:,.2f}%", "Volatilidade": "{:,.2f}%"}),
                 width=800,
                 height=250,
                 column_order=("Fundo", "CAGR", "Volatilidade"),
                 hide_index=True)

    jm_tamanho = st.slider("Qual é o tamanho da janela móvel desejada?",
                           min_value=21,
                           max_value=252,
                           value=21*6)
    jm_volatilidade = retornos.rolling(
        jm_tamanho).apply(ar.calcular_vol).multiply(100)
    jm_volatilidade = jm_volatilidade.dropna()
    fig_vol = go.Figure()
    for coluna in jm_volatilidade.columns:
        fig_vol.add_trace(go.Scatter(x=jm_volatilidade.index,
                                     y=jm_volatilidade[coluna],
                                     mode="lines",
                                     name=coluna[0:15]))

    fig_vol.update_layout(
        title="Volatilidade dos Fundos",
        xaxis_title="Data",
        yaxis_title="Vol",
        hovermode='x',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='right',
            x=1))

    st.plotly_chart(fig_vol)


if pagina_select == "Drawdown":
    drawdown = cotas.apply(ar.calcular_drawdown_hist).multiply(100)
    drawdown = drawdown.round(2)

    fig = go.Figure()
    for coluna in drawdown.columns:
        fig.add_trace(go.Scatter(x=drawdown.index,
                                 y=drawdown[coluna],
                                 mode="lines",
                                 name=coluna))
    fig.update_layout(title="Drawdown dos Fundos",
                      xaxis_title="Data",
                      yaxis_title="Drawdown",
                      hovermode='x',
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  y=1,
                                  xanchor='right',
                                  x=1))
    st.plotly_chart(fig)


if pagina_select == "VaR":
    ic = st.slider("Qual é o Intervalo de Confiança?",
                   min_value=90.0,
                   max_value=100.0,
                   value=95.0)

    for coluna in retornos.columns:
        var_hist = ar.calcular_var_hist(retornos[coluna], alpha=ic)*100
        var_param = ar.calcular_var_param(retornos[coluna], alpha=ic)*100
        fig = px.histogram(retornos[coluna]*100,
                           x=coluna,
                           nbins=200,
                           histnorm='probability density',
                           title=coluna)

        fig.add_trace(go.Scatter(x=[var_hist, var_hist],
                                 y=[0, 0.3], mode='lines',
                                 name='VaR histórico',
                                 line=dict(color='red',
                                           width=2)))

        fig.add_trace(go.Scatter(x=[var_param, var_param],
                                 y=[0, 0.3],
                                 mode='lines',
                                 name='VaR Paramétrico',
                                 line=dict(color='blue',
                                           width=2)))

        fig.update_layout(xaxis_title='Retorno',
                          yaxis_title='Densidade',
                          showlegend=True)
        st.plotly_chart(fig)

if pagina_select == "Markowitz":
    n_simulacoes = st.slider("Qual o número de portfólios simulados?",
                             min_value=1000,
                             max_value=10000,
                             value=4000,
                             step=1000)

    retornos_no_periodo = cotas.iloc[-1]/cotas.iloc[0]
    cagr_fundos = (
        (retornos_no_periodo**(len(retornos_no_periodo)/252))-1).multiply(100)
    volatilidades_fundos = retornos.apply(ar.calcular_vol)
    cov_fundos = retornos.cov()

    pesos_ports = np.zeros((n_simulacoes, retornos.shape[1]))
    retornos_ports = np.zeros(n_simulacoes)
    volatilidades_ports = np.zeros(n_simulacoes)
    sharpes_ports = np.zeros(n_simulacoes)

    for i_port in range(n_simulacoes):
        pesos = np.array(np.random.random(retornos.shape[1]))
        pesos = pesos/sum(pesos)
        pesos_ports[i_port, :] = pesos

        cagr = np.sum(cagr_fundos*pesos)
        retornos_ports[i_port] = cagr

        vol = ar.calcular_vol_portfolio(pesos, cov_fundos)
        volatilidades_ports[i_port] = vol

        sharpe = cagr/vol
        sharpes_ports[i_port] = sharpe

    fig_fe = go.Figure()
    fig_fe.add_trace(go.Scatter(x=volatilidades_ports*100,
                                y=retornos_ports,
                                mode="markers",
                                name='Portfólios'))
    fig_fe.update_layout(title="Fronteira Eficiente",
                         xaxis_title="Volatilidade",
                         yaxis_title="Retorno",
                         hovermode='x')
    st.plotly_chart(fig_fe)

    max_sharpe_index = sharpes_ports.argmax()
    max_sharpe_portfolio = pesos_ports[max_sharpe_index, :]
    print(max_sharpe_portfolio)
