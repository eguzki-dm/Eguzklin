import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_corr(
    df,
    method='spearman',
    figsize=(11, 9),
    annot=True,
    cmap='coolwarm',
    mask_upper=True,
    center=0,
    linewidths=0.5,
    top_corr_only=False,
    corr_threshold=0.5
):
    """
    Visualiza matriz de correlación para variables numéricas.

    Parameters
    ----------
    df : DataFrame
        Dataset de entrada.

    method : str
        Método de correlación.

    top_corr_only : bool
        Mostrar solo correlaciones fuertes.

    corr_threshold : float
        Umbral mínimo de correlación absoluta.
    """

    # =========================
    # Variables numéricas
    # =========================

    numeric_df = df.select_dtypes(
        include=['number', 'bool']
    )

    # =========================
    # Correlación
    # =========================

    corr = numeric_df.corr(method=method)

    # =========================
    # Filtrar correlaciones fuertes
    # =========================

    if top_corr_only:

        corr_filtered = corr.copy()

        # Mantener solo correlaciones fuertes
        corr_filtered[
            np.abs(corr_filtered) < corr_threshold
        ] = np.nan

        # Eliminar filas/columnas vacías
        corr_filtered = corr_filtered.dropna(
            axis=0,
            how='all'
        )

        corr_filtered = corr_filtered.dropna(
            axis=1,
            how='all'
        )

        corr = corr_filtered

    # =========================
    # Máscara triángulo superior
    # =========================

    mask = None

    if mask_upper:

        mask = np.triu(
            np.ones_like(corr, dtype=bool)
        )

    # =========================
    # Plot
    # =========================

    plt.figure(figsize=figsize)

    sns.heatmap(

        corr,

        mask=mask,

        annot=annot,

        cmap=cmap,

        center=center,

        square=True,

        linewidths=linewidths,

        cbar_kws={"shrink": 0.8}
    )

    plt.title(
        f'Correlation Heatmap ({method.title()})'
    )

    plt.tight_layout()

    plt.show()

    return corr

def heatmap_corr_plot(df):
    sns.set_theme(style="white")

    numeric_df = df.select_dtypes(include=["number", "bool"])

    # Compute the correlation matrix
    corr = numeric_df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}) 
    

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("..\data\hormigon.csv")
    heatmap_corr_plot(df)
    plt.savefig('heatmap.png')