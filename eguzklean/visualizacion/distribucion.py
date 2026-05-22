import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualizar_balance_clases(y):
    '''
    Distribución entre clases TARGET
    '''
    conteo = pd.Series(y).value_counts().sort_index()

    plt.figure(figsize=(8,5))

    sns.barplot(
        x=conteo.index,
        y=conteo.values
    )

    plt.title("Distribución de clases")

    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

    return conteo



def visualizar_distribuciones(
    df,
    bins=30,
    kde=True, # False para datasets grandes
    figsize=(16, 12),
    cols_per_row=3
):
    """
    Visualiza distribuciones de variables numéricas.

    Parameters
    ----------
    df : DataFrame
        Dataset de entrada.

    bins : int
        Número de bins histograma.

    kde : bool
        Mostrar KDE.

    figsize : tuple
        Tamaño figura global.

    cols_per_row : int
        Número columnas subplot.
    """

    # =====================================================
    # Variables numéricas
    # =====================================================

    numeric_df = df.select_dtypes(
        include=['number']
    )

    columns = numeric_df.columns

    n_cols = cols_per_row

    n_rows = math.ceil(
        len(columns) / n_cols
    )

    # =====================================================
    # Figura
    # =====================================================

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize
    )

    # Convertir axes a array plano
    axes = axes.flatten()

    # =====================================================
    # Histogramas
    # =====================================================

    for i, col in enumerate(columns):

        sns.histplot(

            numeric_df[col],

            bins=bins,

            kde=kde,

            ax=axes[i]
        )

        axes[i].set_title(col)

        axes[i].set_xlabel('')

    # =====================================================
    # Eliminar subplots vacíos
    # =====================================================

    for j in range(i + 1, len(axes)):

        fig.delaxes(axes[j])

    # =====================================================
    # Layout
    # =====================================================

    plt.suptitle(
        'Distribución Variables Numéricas',
        fontsize=16
    )

    plt.tight_layout()

    plt.show()