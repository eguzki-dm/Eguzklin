import pandas as pd
import numpy as np
from scipy.stats import shapiro


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
        'Cardinalidad': df.nunique()})
    return resumen


import pandas as pd
import numpy as np
from scipy.stats import shapiro


def describir_datos(df):
    """
    Descripción completa de los datos del dataframe.
    """
    descripcion = df.describe(include='all').T
    descripcion['Nulos'] = df.isnull().sum()
    descripcion['Tipos'] = df.dtypes
    return descripcion


def resumir_datos(df):
    """
    Resumen básico de los datos del dataframe.
    """
    resumen = pd.DataFrame({
        'Tipo': df.dtypes,
        'Nulos': df.isnull().sum(),
        'Duplicados': df.apply(lambda x: x.duplicated().sum()),
        'Cardinalidad': df.nunique()
    })
    return resumen


def super_resumen(df, alpha=0.05, corr_threshold=0.8):
    """
    Genera un resumen estadístico completo del dataset sin imprimir tablas feas.
    Devuelve un diccionario con los DataFrames de resultados y muestra una
    guía de uso en la consola.

    Parameters
    ----------
    df : DataFrame
        Dataset de entrada.
    alpha : float
        Nivel de significancia para el test de normalidad (Shapiro-Wilk).
    corr_threshold : float
        Umbral para filtrar correlaciones fuertes.

    Returns
    -------
    resumen : dict
        Diccionario con los DataFrames de cada sección.
    """

    # 1. INFORMACIÓN GENERAL
    n_rows, n_cols = df.shape
    general_df = pd.DataFrame({
        "Métrica": ["Filas", "Columnas"],
        "Cantidad": [n_rows, n_cols]
    })

    # 2. TIPOS DE DATOS
    dtypes_df = pd.DataFrame({
        'Variable': df.columns,
        'Tipo': df.dtypes.astype(str)
    })

    # 3. VALORES NULOS
    nulls_df = pd.DataFrame({
        'Variable': df.columns,
        'Nulos': df.isnull().sum(),
        '% Nulos': (df.isnull().mean() * 100).round(2)
    }).sort_values(by='Nulos', ascending=False)

    # 4. DUPLICADOS
    duplicated_df = pd.DataFrame({
        "Métrica": ["Filas Duplicadas en el Dataset"],
        "Cantidad": [df.duplicated().sum()]
    })

    # 5. CARDINALIDAD
    cardinalidad_df = pd.DataFrame({
        'Variable': df.columns,
        'Valores únicos': df.nunique()
    }).sort_values(by='Valores únicos', ascending=False)

    # 6. TEST DE NORMALIDAD
    numeric_df = df.select_dtypes(include=['number'])
    normalidad_resultados = []

    for col in numeric_df.columns:
        serie = numeric_df[col].dropna()

        # Límite práctico para el test de Shapiro-Wilk
        if len(serie) > 5000:
            serie = serie.sample(5000, random_state=42)

        try:
            stat, p = shapiro(serie)
            normal = p > alpha
        except:
            p = np.nan
            normal = False

        normalidad_resultados.append({
            'Variable': col,
            'p-value': p,
            'Normal': normal
        })

    normalidad_df = pd.DataFrame(normalidad_resultados)

    # 7. CORRELACIONES FUERTES (Spearman)
    corr = numeric_df.corr(method='spearman')
    
    # Extraer la matriz triangular superior para evitar parejas duplicadas
    corr_pairs = (
        corr.abs()
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    
    corr_pairs.columns = ['Variable_1', 'Variable_2', 'Correlación']
    corr_pairs = corr_pairs[corr_pairs['Correlación'] >= corr_threshold]
    corr_pairs = corr_pairs.sort_values(by='Correlación', ascending=False)

    # Estructurar el diccionario de salida
    resumen = {
        'general': general_df,
        'dtypes': dtypes_df,
        'nulls': nulls_df,
        'duplicated': duplicated_df,
        'cardinality': cardinalidad_df,
        'normality': normalidad_df,
        'strong_correlations': corr_pairs
    }

    # Mensaje guía para el usuario
    print("\n" + "✓ " * 25)
    print("¡Super Resumen procesado con éxito!")
    print("Guarda el resultado en una variable (ej: res = super_resumen(df))")
    print("y visualiza las secciones que necesites:")
    print("  • res['general']             -> Dimensiones del dataset")
    print("  • res['dtypes']              -> Tipos de datos por columna")
    print("  • res['nulls']               -> Cantidad y % de nulos")
    print("  • res['duplicated']          -> Conteo de filas idénticas")
    print("  • res['cardinality']         -> Valores únicos por columna")
    print("  • res['normality']           -> Resultados del Test de Shapiro")
    print("  • res['strong_correlations'] -> Parejas con correlación alta")
    print("\n💡 Consejo en Jupyter/Colab: Usa display(res['nulls']) para tablas interactivas.")
    print("✓ " * 25 + "\n")

    return resumen


if __name__ == "__main__":
    # Bloque de prueba local para cuando ejecutas el archivo directamente en VSC
    print("Ejecutando prueba local...")
    try:
        # Intentamos cargar el archivo usando la ruta relativa del proyecto
        df_prueba = pd.read_csv("data/hormigon.csv")
        res = super_resumen(df_prueba)
    except FileNotFoundError:
        print("Nota: No se encontró 'data/hormigon.csv' para la prueba local.")


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../data/hormigon.csv")
    res= resumir_datos(df)
    print(res)
