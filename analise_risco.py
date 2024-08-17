import numpy as np
import pandas as pd
import scipy


def calcular_vol(retornos: pd.Series) -> float:
    """
    Calcula a volatilidade anualizada de uma série temporal de cotas de um fundo.

    Args:
    retornos (pd.Series): Série temporal dos retornos do fundo.
    """
    vol = retornos.std() * np.sqrt(252)
    return vol


def calcular_vol_portfolio(pesos: np.array, matriz_cov: pd.DataFrame) -> float:
    """
    Calcula a volatilidade anualizada de um portfólio, dado os pesos e a matriz de covariancia.

    Args:
    pesos (np.array): 
    matriz_cov: 
    """
    vol = np.sqrt(pesos.T.dot(matriz_cov).dot(pesos))
    vol = vol * np.sqrt(252)
    return vol


def calcular_var_hist(retornos: pd.Series, alpha: int = 95, days: int = 1) -> float:
    """
    Calcula o Valor em Risco (VaR) histórico de uma série temporal de cotas de um fundo.

    Args:
    retornos (pd.Series): Série temporal dos retornos do fundo.
    alpha (int, opcional): Percentil de confiança (padrão é 95).
    days (int, opcional): Número de dias para o qual o VaR é calculado (padrão é 1).
    """
    value_at_risk = np.percentile(retornos, 100-alpha)
    return value_at_risk * np.sqrt(days)


def calcular_var_param(retornos: pd.Series, alpha: int = 95, days: int = 1) -> float:
    """
    Calcula o Valor em Risco (VaR) paramétrico, utilizando a distribuição normal,
    de uma série temporal de cotas de um fundo.

    Args:
    retornos (pd.Series): Série temporal dos retornos do fundo.
    alpha (int, opcional): Percentil de confiança (padrão é 95).
    days (int, opcional): Número de dias para o qual o VaR é calculado (padrão é 1).
    """
    alpha = 1-(alpha/100)
    mu = retornos.mean()
    sig = retornos.std()
    value_at_risk = scipy.stats.norm.ppf(alpha, mu, sig) * np.sqrt(days)
    return value_at_risk


def calcular_drawdown_hist(cota_fundo: pd.Series) -> pd.Series:
    """
    Calcula o drawdown histórico de uma série temporal de cotas de um fundo.

    Args:
    cota_fundo (pd.Series): Série temporal das cotas do fundo.
    """
    peak = cota_fundo.iloc[0]
    drawdown_hist = []

    for price in cota_fundo:
        if price > peak:
            peak = price
        if peak == price:
            drawdown = 0
        else:
            drawdown = (price / peak) - 1
        drawdown_hist.append(drawdown)

    drawdown_series = pd.Series(drawdown_hist, index=cota_fundo.index)
    return drawdown_series


def monte_carlo_portfolios(retornos: pd.Series, n_simulacoes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Através de milhares de simulações, gera vários portfólios, retornando suas características.

    Args:
    retornos (pd.Series): Série temporal dos retornos do fundo.
    n_simulacoes (int): Número de simulações que deverão ser realizadas.
    """
    retornos_fundos = retornos.mean()
    retornos_fundos = retornos_fundos.add(1).pow(252).subtract(1)
    cov_fundos = retornos.cov()

    pesos_ports = np.zeros((n_simulacoes, retornos.shape[1]))
    retornos_ports = np.zeros(n_simulacoes)
    volatilidades_ports = np.zeros(n_simulacoes)
    sharpes_ports = np.zeros(n_simulacoes)

    for i_port in range(n_simulacoes):
        pesos = np.array(np.random.random(retornos.shape[1]))
        pesos = pesos/sum(pesos)
        pesos_ports[i_port, :] = pesos

        ret = np.sum(retornos_fundos*pesos)
        retornos_ports[i_port] = ret

        vol = calcular_vol_portfolio(pesos, cov_fundos)*100
        volatilidades_ports[i_port] = vol

        sharpe = ret/vol
        sharpes_ports[i_port] = sharpe

    return pesos_ports, retornos_ports, volatilidades_ports, sharpes_ports
