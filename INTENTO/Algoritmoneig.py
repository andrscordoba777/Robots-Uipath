# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:51:46 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
from geopy.distance import geodesic
from scipy.spatial.distance import cdist



"Se cargan los datos dese el arcivo excel"
data = pd.read_excel("C:\\Users\\user\\Desktop\\Prueba.xlsx", header=None)
    
"Se organizan los datos para poder trabajar con ellos"
data = data.drop([1], axis=1)
datos=data.dropna()
datos= np.array(datos)
dim=np.shape(datos)

list_Coord=[]

info_Fin=[]


for i in range(dim[0]):
    txt="(?<=\/\@)[\d\.]+\,[\d\.\-]+"
    b=re.findall(txt,datos[i,8])
    b="".join(b)
    list_Coord.append(b)
    info=datos[i,1]+" - "+datos[i,4]+" - "+datos[i,3]+"\n"+datos[i,7]
    info_Fin.append(info)
    b=re.findall(txt,datos[i,8])
    b="".join(b)
    coord = list(map(float, b.split(',')))
    

# Función para convertir coordenadas geográficas a cartesianas
def geo_to_cartesian(lat, lon):
    R = 6371  # Radio de la Tierra en km
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    return x, y

# Función para calcular la distancia total de una ruta
def calcular_distancia_total(ruta, distancias):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        ciudad_actual = ruta[i]
        ciudad_siguiente = ruta[i + 1]
        distancia_total += distancias[ciudad_actual, ciudad_siguiente]
    return distancia_total

from math import radians, cos, sin, asin, sqrt

def distancia_entre_puntos(lat1, lon1, lat2, lon2):
    """
    Retorna la distancia en kilómetros entre dos puntos dados por su latitud y longitud
    utilizando la fórmula de Haversine.
    """
    # convertir grados a radianes
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # aplicar fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # radio de la Tierra en kilómetros
    return c * r
def vecino_mas_cercano(coordenadas, punto_inicial=None):
    if punto_inicial is not None:
        coordenadas = np.insert(coordenadas, 0, punto_inicial, axis=0)

    # Convertir coordenadas geográficas a cartesianas
    coordenadas_cartesianas = np.array([geo_to_cartesian(lat, lon) for lat, lon in coordenadas])

    # Calcular matriz de distancias
    matriz_distancias = cdist(coordenadas_cartesianas, coordenadas_cartesianas, 'euclidean')

    # Algoritmo del vecino más cercano
    ruta_indices = [0]
    no_visitados = set(range(1, len(coordenadas)))
    while no_visitados:
        indice_actual = ruta_indices[-1]
        distancias = [(i, matriz_distancias[indice_actual][i]) for i in no_visitados]
        indice_mas_cercano, distancia_mas_cercana = min(distancias, key=lambda x: x[1])
        ruta_indices.append(indice_mas_cercano)
        no_visitados.remove(indice_mas_cercano)

    # Obtener la ruta óptima en coordenadas geográficas
    ruta_optima = [coordenadas[i] for i in ruta_indices]

    # Calcular la distancia total de la ruta óptima
    distancia_total = sum([matriz_distancias[ruta_indices[i-1]][ruta_indices[i]] for i in range(1, len(ruta_indices))])

    return ruta_optima, distancia_total, ruta_indices


    
lista_Coord2 = [list(map(float, x.split(','))) for x in list_Coord]
coordenadas = np.array(lista_Coord2)

start_point=[6.259041568722493, -75.59461563223755]
ruta_optima, distancia_optima,x = vecino_mas_cercano(coordenadas, start_point)
ruta_final=ruta_optima[1:]


x=np.array(x)
x=np.delete(x,0)
print("Ruta óptima: ", x)
print("Distancia total en kilómetros: ", distancia_optima)

# Crear arreglo de valores a partir de ruta_final
valores = np.arange(len(ruta_final))+1

# Convertir las coordenadas geográficas a coordenadas cartesianas
coordenadas_cartesianas = np.array([geo_to_cartesian(lat, lon) for lat, lon in ruta_final])
coordenadas_cartesianas[:, [0, 1]] = coordenadas_cartesianas[:, [1, 0]]

# Graficar puntos y trazar la línea del recorrido en coordenadas cartesianas
plt.scatter(coordenadas_cartesianas[:, 1], coordenadas_cartesianas[:, 0], c=valores, cmap='viridis')
plt.plot(coordenadas_cartesianas[:, 1], coordenadas_cartesianas[:, 0], '-o')

# Agregar etiquetas para los puntos
for i, valor in enumerate(valores):
    plt.annotate(valor, (coordenadas_cartesianas[i, 1], coordenadas_cartesianas[i, 0]))

# Configurar los ejes
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar la gráfica
plt.show()

file = open("C:\\Users\\user\\Documents\\UiPath\\INTENTO\\ruta\\ruta2.txt", "w")    
for j in range(len(x)):
    valor=(str(x[j]-1))
    file.write(str(info_Fin[int(valor)-1]) + os.linesep)
file.close()
    
file.close()
