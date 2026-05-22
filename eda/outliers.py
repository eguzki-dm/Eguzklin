import pandas as pd
import numpy as np


# =========================================================
# DETECTAR OUTLIERS - IQR
# =========================================================

def detectar_outliers_iqr(
    df,
    factor=1.5,
    mostrar_solo_con_outliers=True
):
    """
    Detecta outliers usando el método IQR.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de entrada.

    factor : float, default=1.5
        Factor multiplicador del IQR.

    mostrar_solo_con_outliers : bool, default=True
        Si True, muestra únicamente variables
        que contienen outliers.

    Returns
    -------
    pd.DataFrame
        Resumen de outliers por variable.
    """

    # =====================================================
    # Variables numéricas
    # =====================================================

    numeric_df = df.select_dtypes(include=np.number)

    resultados = []

    # =====================================================
    # Evaluar columnas
    # =====================================================

    for col in numeric_df.columns:

        serie = numeric_df[col].dropna()

        # Saltar columnas vacías
        if len(serie) == 0:
            continue

        # Cuartiles
        q1 = serie.quantile(0.25)

        q3 = serie.quantile(0.75)

        iqr = q3 - q1

        # Límites
        lower_bound = q1 - (factor * iqr)

        upper_bound = q3 + (factor * iqr)

        # Outliers
        mask_outliers = (
            (serie < lower_bound) |
            (serie > upper_bound)
        )

        n_outliers = mask_outliers.sum()

        pct_outliers = (
            n_outliers / len(serie)
        ) * 100

        resultados.append({

            'Variable': col,

            'Q1': round(q1, 4),

            'Q3': round(q3, 4),

            'IQR': round(iqr, 4),

            'Lower Bound': round(lower_bound, 4),

            'Upper Bound': round(upper_bound, 4),

            'N Outliers': int(n_outliers),

            '% Outliers': round(
                pct_outliers,
                2
            )
        })

    # =====================================================
    # DataFrame resultados
    # =====================================================

    outliers_df = pd.DataFrame(resultados)

    # Filtrar
    if mostrar_solo_con_outliers:

        outliers_df = outliers_df[
            outliers_df['N Outliers'] > 0
        ]

    # Ordenar
    outliers_df = outliers_df.sort_values(
        by='% Outliers',
        ascending=False
    ).reset_index(drop=True)

    return outliers_df


# =========================================================
# ELIMINAR OUTLIERS - IQR
# =========================================================

def eliminar_outliers_iqr(
    df,
    outliers_df
):
    """
    Elimina outliers usando límites definidos
    en outliers_df generado por detectar_outliers_iqr().

    IMPORTANTE:
    - Conserva valores NaN.
    - Elimina filas únicamente cuando el valor
      está fuera de los límites.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    outliers_df : pd.DataFrame
        Resultado generado por detectar_outliers_iqr().

    Returns
    -------
    pd.DataFrame
        Dataset limpio sin outliers.
    """

    # =====================================================
    # Copia dataset
    # =====================================================

    df_clean = df.copy()

    filas_originales = len(df_clean)

    # =====================================================
    # Filtrar outliers
    # =====================================================

    for _, row in outliers_df.iterrows():

        variable = row['Variable']

        lower = row['Lower Bound']

        upper = row['Upper Bound']

        # Mantener:
        # - valores dentro del rango
        # - NaN

        mask = (

            (
                (df_clean[variable] >= lower) &
                (df_clean[variable] <= upper)
            )

            |

            (df_clean[variable].isna())
        )

        df_clean = df_clean[mask]

    # =====================================================
    # Resetear índice
    # =====================================================

    df_clean = df_clean.reset_index(drop=True)

    # =====================================================
    # Resumen
    # =====================================================

    filas_finales = len(df_clean)

    filas_eliminadas = (
        filas_originales - filas_finales
    )

    pct_eliminado = (
        (filas_eliminadas / filas_originales) * 100
    )

    print("=" * 50)

    print("ELIMINACIÓN DE OUTLIERS")

    print("=" * 50)

    print(f"\nFilas originales : {filas_originales}")

    print(f"Filas finales    : {filas_finales}")

    print(f"Filas eliminadas : {filas_eliminadas}")

    print(f"% eliminado      : {pct_eliminado:.2f}%")

    return df_clean


# =========================================================
# DETECTAR OUTLIERS - Z SCORE
# =========================================================

def detectar_outliers_zscore(
    df,
    threshold=3,
    mostrar_solo_con_outliers=True
):
    """
    Detecta outliers usando Z-Score
    (desviación estándar).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset entrada.

    threshold : float, default=3
        Número de desviaciones estándar
        para considerar un outlier.

    mostrar_solo_con_outliers : bool, default=True
        Mostrar únicamente variables
        con outliers.

    Returns
    -------
    pd.DataFrame
        Resumen de outliers.
    """

    # =====================================================
    # Variables numéricas
    # =====================================================

    numeric_df = df.select_dtypes(include=np.number)

    resultados = []

    # =====================================================
    # Evaluar columnas
    # =====================================================

    for col in numeric_df.columns:

        serie = numeric_df[col].dropna()

        # Saltar columnas vacías
        if len(serie) == 0:
            continue

        mean = serie.mean()

        std = serie.std()

        # Evitar división por cero
        if std == 0:
            continue

        # Z-score
        z_scores = (
            (serie - mean) / std
        )

        # Outliers
        mask_outliers = (
            np.abs(z_scores) > threshold
        )

        n_outliers = mask_outliers.sum()

        pct_outliers = (
            n_outliers / len(serie)
        ) * 100

        resultados.append({

            'Variable': col,

            'Mean': round(mean, 4),

            'Std': round(std, 4),

            'Threshold': threshold,

            'Lower Bound': round(
                mean - (threshold * std),
                4
            ),

            'Upper Bound': round(
                mean + (threshold * std),
                4
            ),

            'N Outliers': int(n_outliers),

            '% Outliers': round(
                pct_outliers,
                2
            )
        })

    # =====================================================
    # DataFrame resultados
    # =====================================================

    outliers_df = pd.DataFrame(resultados)

    # Filtrar
    if mostrar_solo_con_outliers:

        outliers_df = outliers_df[
            outliers_df['N Outliers'] > 0
        ]

    # Ordenar
    outliers_df = outliers_df.sort_values(
        by='% Outliers',
        ascending=False
    ).reset_index(drop=True)

    return outliers_df


# =========================================================
# ELIMINAR OUTLIERS - Z SCORE
# =========================================================

def eliminar_outliers_zscore(
    df,
    outliers_df
):
    """
    Elimina outliers usando límites
    definidos por Z-Score.

    Conserva valores NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    outliers_df : pd.DataFrame
        Resultado generado por
        detectar_outliers_zscore().

    Returns
    -------
    pd.DataFrame
        Dataset limpio.
    """

    # =====================================================
    # Copia dataset
    # =====================================================

    df_clean = df.copy()

    filas_originales = len(df_clean)

    # =====================================================
    # Filtrar variables
    # =====================================================

    for _, row in outliers_df.iterrows():

        variable = row['Variable']

        lower = row['Lower Bound']

        upper = row['Upper Bound']

        # Mantener:
        # - valores dentro rango
        # - NaN

        mask = (

            (
                (df_clean[variable] >= lower) &
                (df_clean[variable] <= upper)
            )

            |

            (df_clean[variable].isna())
        )

        df_clean = df_clean[mask]

    # =====================================================
    # Reset index
    # =====================================================

    df_clean = df_clean.reset_index(drop=True)

    # =====================================================
    # Resumen
    # =====================================================

    filas_finales = len(df_clean)

    filas_eliminadas = (
        filas_originales - filas_finales
    )

    pct_eliminado = (
        (filas_eliminadas / filas_originales) * 100
    )

    print("=" * 50)

    print("ELIMINACIÓN OUTLIERS Z-SCORE")

    print("=" * 50)

    print(f"\nFilas originales : {filas_originales}")

    print(f"Filas finales    : {filas_finales}")

    print(f"Filas eliminadas : {filas_eliminadas}")

    print(f"% eliminado      : {pct_eliminado:.2f}%")

    return df_clean