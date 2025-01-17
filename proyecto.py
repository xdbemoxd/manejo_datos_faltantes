import janitor
import matplotlib.pyplot as plt
import missingno 
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns
import session_info
import upsetplot
import funciones_pandas as fp

df = pd.read_csv("books_1.Best_Books_Ever.csv")

df_aux = df['price']

aux_np = df_aux.to_numpy()

lista = []

for aux in aux_np:
    if str(aux).count(".") == 2 or aux == "":
        lista.append(aux)

df_aux = pd.DataFrame(aux_np, columns=['price'])

df_aux = df_aux.replace( lista, np.nan )

df_aux['price'] = df_aux['price'].astype('float')

df['price'] = df_aux['price']

print(df.dtypes)

#visualizamos primeramente para ver la cantadad de dimensiones que tiene este dataset
#print(df.shape)

#aqui vizualizamos la cantidad de datos faltantes en el dataset
#print(fp.number_missing(df))

#veamos donde estan realmente los valores NULL en el dataset
#print(fp.missing_variable_summary(df))

#acontinuacion seleccione las 3 variables con más casos faltantes
"""fp.missing_variable_span_plot(
    df,
    "edition",
    5400,
    0
)

fp.missing_variable_span_plot(
    df,
    "series",
    5400,
    0
)

fp.missing_variable_span_plot(
    df,
    "firstPublishDate",
    5400,
    0
)"""

#verifico si es un caso MCAR, MAR ó MNAR

################### MCAR #######################

"""fp.sort_variables_by_missingness(df).pipe(missingno.matrix)
plt.show()"""

"""con esto puedo concluir que hay muchos datos perdidos y se relacionan mucho series con edition, procedere a analizar con MAR, 
teniendo en cuenta que edition es la variable con más casos perdidos"""

################## MAR #########################

"""fp.sort_variables_by_missingness(df).sort_values(by="edition").pipe(missingno.matrix)
plt.show()"""    

#la variable edition no se relaciona con otra variable en cuanto a perdida de datos, probare con series

"""fp.sort_variables_by_missingness(df).sort_values(by="series").pipe(missingno.matrix)
plt.show()"""   

#sigue estando muy sesgada la relacion, usare la rercera variable firstPublishDate, a ver si hay alguna relación 

"""fp.sort_variables_by_missingness(df).sort_values(by="firstPublishDate").pipe(missingno.matrix)
plt.show()"""

#no encuentro relación, pero por si acaso usare la cuarta variable con más casos faltantes price

"""fp.sort_variables_by_missingness(df).sort_values(by="price").pipe(missingno.matrix)
plt.show()"""

#no existe relación alguna entre las variables, no creo que sea un caso mnar, porque estamos hablando de libros 
#y los casos son muy que deben tener muy poco parentezco

"""usaremos la matriz de sombra para verificar la veracidad de los datos"""

"""
df_bool = df.isna().replace(
    {
        False: "Not missing",
        True: "Missing"
    }
).add_suffix("_NA").pipe(
    lambda shadow_matrix : pd.concat(
        [df,shadow_matrix],
        axis="columns"
    )
)

fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.displot(
            data=df,
            x='rating',
            hue='edition_NA',
            kind='kde'
        )
    )
)

plt.show()

fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.displot(
            data=df,
            x='rating',
            hue='series_NA',
            kind='kde'
        )
    )
)

plt.show()

fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.displot(
            data=df,
            x='rating',
            hue='firstPublishDate_NA',
            kind='kde'
        )
    )
)

plt.show()
"""

"""comparando cada variable con respecto al raiting, se observa que existe una gran cantidad de valores faltantes en los puntos fuertes del raiting, o sea, 
el raiting en 4 alcanza su pico más alto, pero en edition, es donde hay excivamente más perdida de datos, al punto que llega a ser más alto la cantidad de datos perdidos
a los datos que estan explicitos"""


"""a continuación podemos observar la matriz de correlación, se puede observar que las variables no tiene ninguna relacion con respecto a la perdida de datos"""

"""
missingno.heatmap(
    df=df
)
plt.show()
"""

#imputacion con base a un valor unico, solo a variables numericas, estas mismas no tienen valores nulos

test = df.select_columns("rating","numRatings", "likedPercent", "price")

#Una variable
"""
fp.bind_shadow_matrix(df=test, true_string=True,false_string=False).apply(
    axis='rows',
    func = lambda column: column.fillna(column.mean()) if '_NA' not in column.name else column
).pipe(
    lambda df:(
        sns.displot(
            data=df,
            x="price",
            hue="price_NA"
        )
    )
)
plt.show()
"""

"""aqui se aprecia la cantidad de datos nulos que tiene la variable price, todos quedaron en el centro"""

#se realizara una evaluaciòn a la variable price y likedPercent porque son las que contienen valores nulos

"""
fp.bind_shadow_matrix(df=test, true_string=True,false_string=False).apply(
    axis='rows',
    func = lambda column: column.fillna(column.mean()) if '_NA' not in column.name else column
).assign(
    imputed = lambda df : df.price_NA | df.likedPercent_NA
).pipe(
     lambda df:(
        sns.scatterplot(
            data=df,
            x="price",
            y="likedPercent",
            hue="imputed"
        )
    )
)
plt.show()
"""

"""viendo esto, puedo ver que los datos faltantes esta pocisionados de forma arbitraria en todo el dataset, no se rigen por una norma"""


"""adicionare unos boxenplot para terminar de hacer un analisis"""

fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.boxenplot(
            data=df,
            x='edition_NA',
            y='rating'
        )
    )
)

plt.show()

fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.boxenplot(
            data=df,
            x='series_NA',
            y='rating'
        )
    )
)

plt.show()


fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.boxenplot(
            data=df,
            x='firstPublishDate_NA',
            y='rating'
        )
    )
)

plt.show()


fp.bind_shadow_matrix(df,only_missing=True).pipe(
    lambda df : (
        sns.boxenplot(
            data=df,
            x='price_NA',
            y='rating'
        )
    )
)

plt.show()

"""despues de todo este analisis, sigo con la creencia que los datos estan perdidos de manera aleatoria, quizas al creador del dataset se le paso agregar algunos datos,
al parecer no siguen un patron"""
