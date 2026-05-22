import math
import matplotlib.pyplot as plt
import seaborn as sns


def visualizar_boxplots(
    df,
    figsize=(16, 12),
    cols_per_row=3
):
    """
    Visualiza boxplots de variables numéricas.

    Parameters
    ----------
    df : DataFrame
        Dataset de entrada.

    figsize : tuple
        Tamaño figura.

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

    axes = axes.flatten()

    # =====================================================
    # Boxplots
    # =====================================================

    for i, col in enumerate(columns):

        sns.boxplot(

            x=numeric_df[col],

            ax=axes[i],

            color='#C8A2C8'   # Lila suave
        )

        axes[i].set_title(

            col,

            fontsize=11
        )

        axes[i].set_xlabel('')

        axes[i].grid(
            alpha=0.3
        )

    # =====================================================
    # Eliminar subplots vacíos
    # =====================================================

    for j in range(i + 1, len(axes)):

        fig.delaxes(axes[j])

    # =====================================================
    # Layout
    # =====================================================

    plt.suptitle(
        'Boxplots Variables Numéricas',
        fontsize=16
    )

    plt.tight_layout()

    plt.show()