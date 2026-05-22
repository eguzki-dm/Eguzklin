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


