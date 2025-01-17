# manejo_datos_faltantes
proyecto que realizare a un dataset, para ver la frecuencias de los datos faltantes 

## debes instalar las biblotecas puestas en el archivo requirements.txt

'''
introduzca en la terminal el comando 

pip install -r requirements.txt
'''

## primero que nada debes descargar el dataset de datos, esta en el siguiente link 

'''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("pooriamst/best-books-ever-dataset")

print("Path to dataset files:", path)
'''

## una vez cargado el dataset, podemos comenzar con el analisis, seleccione las 3 columnas con más valores faltantes

'''
MCAR 

    Falta completamente al azar
    La probabilidad de que falten valores no está relacionada con el valor de la variable
    Se puede analizar la observación observada e ignorar las faltantes 

MAR 

    Desaparecido al azar
    La probabilidad de que falten datos está relacionada con otras variables 

MNAR 

    Desaparecido no al azar
    La probabilidad de que falten datos depende de los valores de la variable
'''