import pandas as pd
import numpy as np
import itertools
import upsetplot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


#cantidad de datos faltantes en un dataset
def number_missing(df) -> int:
    return df.isna().sum().sum()

#cantidad de datos en total sin perdidas
def number_complete(df) -> int:
    return df.size - number_missing(df)

#una matriz donde se ve la cantidad de valores completos, NULL y el porcentaje de NULL
def missing_variable_summary(df) -> pd.DataFrame:
    return df.isnull().pipe(
        lambda df_1: (
            df_1.sum()
            .reset_index(name="n_missing")
            .rename(columns={"index": "variable"})
            .assign(
                n_cases=len(df_1),
                pct_missing=lambda df_2: df_2.n_missing / df_2.n_cases * 100,
            )
        )
    )

"""mide la cantidad de datos faltantes segun la fila, en esa fila ve cuantas columnas no tienen datos y da un porcentaje con respecto a la fila"""
def missing_case_summary(df) -> pd.DataFrame:
    return df.assign(
        case=lambda df_1: df_1.index,
        n_missing=lambda df_1: df_1.apply(
            axis="columns", func=lambda row: row.isna().sum()
        ),
        pct_missing=lambda df_1: df_1["n_missing"] / df_1.shape[1] * 100,
    )[["case", "n_missing", "pct_missing"]]

"""Esta tabla explica lo siguiente:
Primera columna: En esta columna se puede visualizar la cantidad de datos faltantes de un variable (Columna)
Segunda columna: Aqui se observa la cantidad de variables que comparten el mismo numero de datos faltantes
Tercera columna: El porcentaje de columnas con ese misma cantidad de datos faltantes"""
def missing_variable_table(df) -> pd.DataFrame:
        return (
            missing_variable_summary(df)
            .value_counts("n_missing")
            .reset_index(name="n_variables")
            .rename(columns={"n_missing": "n_missing_in_variable"})
            .assign(
                pct_variables=lambda df_1: df_1.n_variables / df_1.n_variables.sum() * 100
            )
            .sort_values("pct_variables", ascending=False)
        ) 

"""Esta tabla explica lo siguiente:
Primera columna: En esta columna se puede visualizar la cantidad de datos faltantes de un variable (Columna)
Segunda columna: Aqui se observa la cantidad de casos que comparten el mismo numero de datos faltantes
Tercera columna: El porcentaje de casos con ese misma cantidad de datos faltantes"""
def missing_case_table(df) -> pd.DataFrame:
        return (
            missing_case_summary(df)
            .value_counts("n_missing")
            .reset_index(name="n_cases")
            .rename(columns={"n_missing": "n_missing_in_case"})
            .assign(
                  pct_case=lambda df_1: df_1.n_cases / df_1.n_cases.sum() * 100
                  )
            .sort_values("pct_case", ascending=False)
        )

"""te separa el dataset por bloques y te dice en cada bloque los espacios vacios que hay, el porcentaje de cada bloque en valores nulos 
y valores llenos """
def missing_variable_span(df, variable: str, span_every: int) -> pd.DataFrame:
        return (
            df.assign(
                span_counter = lambda df_1: (
                    np.repeat(a=range(df_1.shape[0]), repeats=span_every)[: df_1.shape[0]]
                )
            )
            .groupby("span_counter")
            .aggregate(
                n_in_span=(variable, "size"),
                n_missing=(variable, lambda s: s.isnull().sum()),
            )
            .assign(
                n_complete=lambda df_1: df_1.n_in_span - df_1.n_missing,
                pct_missing=lambda df_1: df_1.n_missing / df_1.n_in_span * 100,
                pct_complete=lambda df_1: 100 - df_1.pct_missing,
            )
            .drop(columns=["n_in_span"])
            .reset_index()
        )

#Expresa la cantidad de valores seguidos que hay nulos y completos
def missing_variable_run(df, variable) -> pd.DataFrame:
    rle_list = df[variable].pipe(
        lambda s: [[len(list(g)), k] for k, g in itertools.groupby(s.isnull())]
    )

    return pd.DataFrame(data=rle_list, columns=["run_length", "is_na"]).replace(
        {False: "complete", True: "missing"}
    )

#ordena de la columna con más valores faltantes a la columna que menos tiene
def sort_variables_by_missingness(df, ascending = False):

    return (
            df
            .pipe(
                lambda df_1: (
                    df_1[df_1.isna().sum().sort_values(ascending = ascending).index]
                )
            )
        )

def create_shadow_matrix(
        df,
        true_string: str = "Missing",
        false_string: str = "Not Missing",
        only_missing: bool = False,
    ) -> pd.DataFrame:
        return (
            df
            .isna()
            .pipe(lambda df_1: df_1[df_1.columns[df_1.any()]] if only_missing else df_1)
            .replace({False: false_string, True: true_string})
            .add_suffix("_NA")
        )

#regresa una matriz con columnas_NA las cuales diran si el valor falta o no en la matriz original
#igualmente la matriz original esta ahi tambien
def bind_shadow_matrix(
        df,
        true_string: str = "Missing",
        false_string: str = "Not Missing",
        only_missing: bool = False,
    ) -> pd.DataFrame:
        return pd.concat(
            objs=[
                df,
                create_shadow_matrix(df,
                    true_string=true_string,
                    false_string=false_string,
                    only_missing=only_missing
                )
            ],
            axis="columns"
        )

def missing_scan_count(df, search) -> pd.DataFrame:
    return (
        df.apply(axis="rows", func=lambda column: column.isin(search))
        .sum()
        .reset_index()
        .rename(columns={"index": "variable", 0: "n"})
        .assign(original_type=df.dtypes.reset_index()[0])
    )

# Plotting functions ---

#muestra con un grafico de barras horizontal la cantidad de datos nulos tiene dada columna
def missing_variable_plot(df):
        df_1 = missing_variable_summary(df).sort_values("n_missing")

        plot_range = range(1, len(df_1.index) + 1)

        plt.hlines(y=plot_range, xmin=0, xmax=df_1.n_missing, color="black")

        plt.plot(df_1.n_missing, plot_range, "o", color="black")

        plt.yticks(plot_range, df_1.variable)

        plt.grid(axis="y")

        plt.xlabel("Number missing")
        plt.ylabel("Variable")
        plt.show()

"""grafica donde muestra:
eje x: los casos en donde faltan variables en cada fila
eje y: la cantidad de veces que se repite esto"""
def missing_case_plot(df):

        df_1 = missing_case_summary(df)

        sns.displot(data=df_1, x="n_missing", binwidth=1, color="black")

        plt.grid(axis="x")
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")

        plt.show()

"""muestra el porcentaje de valores nulos que hay en una variable, sepando por bloques que seleccionas tu"""
def missing_variable_span_plot(
        df, variable: str, span_every: int, rot: int = 0, figsize=None
    ):

        (
            missing_variable_span(
                df,variable=variable, span_every=span_every
            ).plot.bar(
                x="span_counter",
                y=["pct_missing", "pct_complete"],
                stacked=True,
                width=1,
                color=["black", "lightgray"],
                rot=rot,
                figsize=figsize,
            )
        )

        plt.xlabel("Span number")
        plt.ylabel("Percentage missing")
        plt.legend(["Missing", "Present"])
        plt.title(
            f"Percentage of missing values\nOver a repeating span of { span_every } ",
            loc="left",
        )
        plt.grid(False)
        plt.margins(0)
        plt.tight_layout(pad=0)
        plt.show()

"""se utiliza para ver las variables que se relacionan con valores faltantes"""
def missing_upsetplot(df, cols: list[str] = None, **kwargs):
    #pandas version -> 2.2.1
    #upsetplot version -> 0.9.0
    if cols is None:
        cols = df.columns.tolist()  # se convierten 'Series' a una lista

    # null values values count
    missing_data = df.isna().value_counts(subset=cols)
        
    # remove the FutureWarnings
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # upsetplot -> library
        upsetplot.plot(missing_data, **kwargs)
        plt.show()