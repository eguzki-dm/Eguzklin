import pandas as pd
from IPython.display import display
from scipy.stats import shapiro
import pandas as pd
import numpy as np

def describir_datos(df):
    '''
    Descripción completa de los datos del dataframe
    '''
    descripcion = df.describe(include='all').T
    descripcion['Nulos'] = df.isnull().sum()
    descripcion['Tipos'] = df.dtypes
    return descripcion

def resumir_datos(df):
    '''
    Resumen de los datos del dataframe:
        Tipo: Tipo de dato
        Nulo: Cantidad de nulos
        Duplicados: Cantidad de duplicados
        Cardinalidad: Cantidad de valores unicos
    '''
    resumen = pd.DataFrame({
        'Tipo': df.dtypes,
        'Nulos': df.isnull().sum(),
        'Duplicados': df.duplicated().sum(),
        'Cardinalidad': df.nunique()
    })
    return resumen


def super_resumen(
    df,
    alpha=0.05,
    corr_threshold=0.8
):
    """
    Genera un resumen estadístico completo del dataset.

    Incluye:
    - tipos de datos
    - nulos
    - duplicados
    - cardinalidad
    - normalidad
    - correlaciones fuertes

    Parameters
    ----------
    df : DataFrame
        Dataset de entrada.

    alpha : float
        Nivel significancia test normalidad.

    corr_threshold : float
        Umbral correlación fuerte.

    Returns
    -------
    resumen : dict
        Diccionario con resultados.
    """

    # =====================================================
    # INFORMACIÓN GENERAL
    # =====================================================

    n_rows, n_cols = df.shape

    print("=" * 60)
    print("RESUMEN ESTADÍSTICO DATASET")
    print("=" * 60)

    print(f"\nFilas: {n_rows}")
    print(f"Columnas: {n_cols}")

    # =====================================================
    # TIPOS DE DATOS
    # =====================================================

    print("\nTIPOS DE DATOS")
    print("-" * 40)

    dtypes_df = pd.DataFrame({

        'Variable': df.columns,

        'Tipo': df.dtypes.astype(str)
    })

    display(dtypes_df)

    # =====================================================
    # NULOS
    # =====================================================

    print("\nVALORES NULOS")
    print("-" * 40)

    nulls_df = pd.DataFrame({

        'Variable': df.columns,

        'Nulos': df.isnull().sum(),

        '% Nulos': (
            df.isnull().mean() * 100
        ).round(2)
    })

    nulls_df = nulls_df.sort_values(
        by='Nulos',
        ascending=False
    )

    display(nulls_df)

    # =====================================================
    # DUPLICADOS
    # =====================================================

    print("\nDUPLICADOS")
    print("-" * 40)

    duplicated = df.duplicated().sum()

    print(f"Duplicados: {duplicated}")

    # =====================================================
    # CARDINALIDAD
    # =====================================================

    print("\nCARDINALIDAD")
    print("-" * 40)

    cardinalidad_df = pd.DataFrame({

        'Variable': df.columns,

        'Valores únicos': df.nunique()
    })

    cardinalidad_df = cardinalidad_df.sort_values(
        by='Valores únicos',
        ascending=False
    )

    display(cardinalidad_df)

    # =====================================================
    # NORMALIDAD
    # =====================================================

    print("\nTEST NORMALIDAD")
    print("-" * 40)

    numeric_df = df.select_dtypes(
        include=['number']
    )

    normalidad_resultados = []

    for col in numeric_df.columns:

        serie = numeric_df[col].dropna()

        # Shapiro límite práctico
        if len(serie) > 5000:

            serie = serie.sample(
                5000,
                random_state=42
            )

        try:

            stat, p = shapiro(serie)

            normal = p > alpha

        except:

            stat = np.nan
            p = np.nan
            normal = False

        normalidad_resultados.append({

            'Variable': col,

            'p-value': p,

            'Normal': normal
        })

    normalidad_df = pd.DataFrame(
        normalidad_resultados
    )

    display(normalidad_df)

    # =====================================================
    # CORRELACIONES FUERTES
    # =====================================================

    print("\nCORRELACIONES FUERTES")
    print("-" * 40)

    corr = numeric_df.corr(method='spearman')

    corr_pairs = (

        corr.abs()

        .where(
            np.triu(
                np.ones(corr.shape),
                k=1
            ).astype(bool)
        )

        .stack()

        .reset_index()
    )

    corr_pairs.columns = [

        'Variable_1',

        'Variable_2',

        'Correlación'
    ]

    corr_pairs = corr_pairs[
        corr_pairs['Correlación'] >= corr_threshold
    ]

    corr_pairs = corr_pairs.sort_values(
        by='Correlación',
        ascending=False
    )

    display(corr_pairs)

    # =====================================================
    # RETURN
    # =====================================================

    resumen = {

        'dtypes': dtypes_df,
        'nulls': nulls_df,
        'cardinality': cardinalidad_df,
        'normality': normalidad_df,
        'strong_correlations': corr_pairs
    }

    return resumen
