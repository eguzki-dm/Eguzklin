# Eguzklean

Funciones auxiliares para EDA, visualización y preprocesamiento.

## Instalación

### Clonando el repositorio

```bash
git clone https://github.com/eguzki-dm/eguzklean.git
cd Eguzklean
pip install .
Instalación directa desde GitHub
pip install git+https://github.com/eguzki-dm/eguzklean.git

## Módulos

### 'eda'

| Función | Descripción |
|---------|-------------|
| `describir_datos(df)` | Descripción completa del DataFrame |
| `resumir_datos(df)` | Resumen con tipos, nulos, duplicados y cardinalidad |
| `super_resumen(df)` | Matriz unificada con dtypes, nulos, cardinalidad, normalidad y correlaciones fuertes |
| `test_normalidad(df)` | Test de Shapiro-Wilk sobre variables numéricas |
| `detectar_outliers_iqr(df)` | Detecta outliers usando rango intercuartílico |
| `eliminar_outliers_iqr(df, outliers_df)` | Elimina outliers según resultados IQR |
| `detectar_outliers_zscore(df)` | Detecta outliers usando Z-Score |
| `eliminar_outliers_zscore(df, outliers_df)` | Elimina outliers según resultados Z-Score |

### 'visualizacion'

| Función | Descripción |
|---------|-------------|
| `heatmap_corr(df)` | Mapa de calor de correlaciones |
| `heatmap_corr_plot(df)` | Versión simplificada del heatmap |
| `visualizar_balance_clases(y)` | Distribución de clases target |
| `visualizar_distribuciones(df)` | Histogramas de variables numéricas |
| `par_real_predicho(y_test, y_pred)` | Gráfico real vs predicho con bisectriz |
| `par_real_predicho_residuos(y_test, y_pred)` | Diagnóstico de residuos con KDE y normal teórica |

## Dependencias

pandas, numpy, scipy, matplotlib, seaborn
```
