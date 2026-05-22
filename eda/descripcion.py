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
    Genera una ÚNICA matriz unificada con dtypes, nulos, cardinalidad y normalidad
    por variable. Detecta automáticamente si estás en un cuaderno (Colab/Jupyter)
    para usar display(), o en un script clásico (VS Code) para retornar los datos.

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
    unificado : DataFrame (o tupla de DataFrames si hay correlaciones)
        La matriz resumen unificada de todas las variables.
    """
    # 1. INFORMACIÓN EN LA CONSOLA
    n_rows, n_cols = df.shape
    duplicados_totales = df.duplicated().sum()
    
    print("=" * 60)
    print("                 ESTADO GLOBAL DEL DATASET")
    print("=" * 60)
    print(f"• Filas totales: {n_rows}")
    print(f"• Columnas totales: {n_cols}")
    print(f"• Filas completamente duplicadas: {duplicados_totales}")
    print("-" * 60)

    # 2. CÁLCULO DE MÉTRICAS BÁSICAS POR COLUMNA
    df_metrics = pd.DataFrame({
        'Tipo': df.dtypes.astype(str),
        'Nulos': df.isnull().sum(),
        '% Nulos': (df.isnull().mean() * 100).round(2),
        'Valores Únicos': df.nunique()
    })

    # 3. TEST DE NORMALIDAD (Solo variables numéricas)
    numeric_df = df.select_dtypes(include=['number'])
    normalidad_resultados = {}

    for col in numeric_df.columns:
        serie = numeric_df[col].dropna()
        if len(serie) > 5000:
            serie = serie.sample(5000, random_state=42)
        try:
            stat, p = shapiro(serie)
            normal = "Sí" if p > alpha else "No"
            p_val = round(p, 4)
        except:
            p_val = np.nan
            normal = "No"
        normalidad_resultados[col] = {'p-value (Shapiro)': p_val, 'Normal': normal}

    df_normalidad = pd.DataFrame.from_dict(normalidad_resultados, orient='index')

    # 4. UNIFICACIÓN EN UNA SOLA TABLA
    unificado = df_metrics.merge(df_normalidad, left_index=True, right_index=True, how='left')
    unificado.index.name = 'Variable'
    unificado = unificado.reset_index()
    
    # Rellenar los huecos de las variables no numéricas (categóricas, fechas...)
    unificado['p-value (Shapiro)'] = unificado['p-value (Shapiro)'].fillna('-')
    unificado['Normal'] = unificado['Normal'].fillna('-')
    unificado = unificado.sort_values(by='Nulos', ascending=False)

    # 5. CÁLCULO DE CORRELACIONES FUERTES (Spearman)
    corr = numeric_df.corr(method='spearman')
    corr_pairs = (
        corr.abs()
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ['Variable_1', 'Variable_2', 'Correlación']
    corr_pairs = corr_pairs[corr_pairs['Correlación'] >= corr_threshold]
    corr_pairs = corr_pairs.sort_values(by='Correlación', ascending=False)

    # 6. DETECCIÓN AUTOMÁTICA DE ENTORNO (Colab/Jupyter vs Terminal)
    try:
        # Si esto no da error, estamos en Google Colab o un Jupyter Notebook
        get_ipython()
        
        print("\n=== MATRIZ RESUMEN DE VARIABLES ===")
        display(unificado)
        
        print("\n=== CORRELACIONES FUERTES DETECTADAS ===")
        if not corr_pairs.empty:
            display(corr_pairs)
        else:
            print("No se detectaron correlaciones que superen el umbral.")
            
    except NameError:
        # Si da error, estamos en un script clásico de VS Code / Terminal
        # No hacemos display(), simplemente dejamos que el desarrollador use el return
        pass

    # Retornamos los objetos por si quieres guardarlos en variables
    if corr_pairs.empty:
        return unificado
    return unificado, corr_pairs


if __name__ == "__main__":
    # Bloque de prueba local para cuando ejecutas el archivo directamente en VSC
    print("Ejecutando prueba local...")
    try:
        df_prueba = pd.read_csv("data/hormigon.csv")
        super_resumen(df_prueba)
    except FileNotFoundError:
        pass