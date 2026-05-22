import pandas as pd
from scipy.stats import shapiro


def test_normalidad(
    df,
    alpha=0.05,
    max_sample=5000):
    """
    Realiza test de normalidad Shapiro-Wilk
    sobre variables numéricas del DataFrame.

    Parameters
    ----------
    df : DataFrame
        Dataset de entrada.
    alpha : float, default=0.05
        Nivel de significancia.
    max_sample : int, default=5000
        Tamaño máximo de muestra para Shapiro.
        (Shapiro no recomienda >5000)

    Returns
    -------
    resultados_df : DataFrame
        Tabla resultados normalidad.
    """

    # =========================
    # Variables numéricas
    # =========================

    numeric_df = df.select_dtypes(
        include=['number'])
    resultados = []

    # =========================
    # Evaluar variables
    # =========================

    for col in numeric_df.columns:
        serie = numeric_df[col].dropna()
        # Limitar tamaño muestra
        if len(serie) > max_sample:
            serie = serie.sample(
                max_sample,
                random_state=42)

        # =========================
        # Shapiro-Wilk
        # =========================

        stat, p_value = shapiro(serie)

        normal = p_value > alpha

        resultados.append({

            'Variable': col,

            'Statistic': stat,

            'p-value': p_value,

            'Normal': normal
        })

    # =========================
    # Resultados
    # =========================

    resultados_df = pd.DataFrame(resultados)

    resultados_df = resultados_df.sort_values(
        by='p-value'
    ).reset_index(drop=True)

    return resultados_df