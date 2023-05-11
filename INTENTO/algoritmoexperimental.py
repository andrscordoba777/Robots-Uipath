# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:02:07 2022

@author: user
"""

import pandas as pd
import numpy as np
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import os
import matplotlib.pyplot as plt



app = Nominatim(user_agent="tycgis")

"Se cargan los datos dese el arcivo excel"
data = pd.read_excel("C:\\Users\\user\\Desktop\\Prueba2.xlsx", header=None)
    
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
    
    



    
lista_Coord2 = [list(map(float, x.split(','))) for x in list_Coord]
lista_Coord2 = np.array(lista_Coord2)

info_Fin=np.array(info_Fin)
list_Coord=np.array(list_Coord)
puntopartida=("6.259041568722493, -75.59461563223755") #punto de partida


mini=[]
ruta=[]
mini2=[]

ruta_completa = []  # lista para almacenar la ruta final

while(len(list_Coord)) > 0:
    for i in range(len(list_Coord)):
        distancia = geodesic(puntopartida, list_Coord[i])
        mini.append(distancia)
        mini2.append(distancia.kilometers)
        
    min_value = min(mini)
    ruta.append(info_Fin[mini.index(min_value)])
    ruta_completa.append(list_Coord[mini.index(min_value)])  # guardar el valor eliminado en la ruta completa
    puntopartida = list_Coord[mini.index(min_value)]
    list_Coord = np.delete(list_Coord, (mini.index(min_value)))
    info_Fin = np.delete(info_Fin, (mini.index(min_value)))
    mini.clear()

ruta_completa=np.array(ruta_completa)
ruta_completa = np.char.split(ruta_completa, sep=',')

# Convertir el resultado a un array de dos columnas
ruta_completa = np.array(ruta_completa.tolist())
ruta_final = ruta_completa.astype(np.float64)

valores = range(1, len(ruta_final) + 1)  # enumeración del orden

# Graficar puntos y trazar la línea del recorrido
plt.scatter(ruta_final[:, 1], ruta_final[:, 0], c=valores, cmap='viridis')
plt.plot(ruta_final[:, 1], ruta_final[:, 0], '-o')

# Agregar etiquetas para los puntos
for i, valor in enumerate(valores):
    plt.annotate(valor, (ruta_final[i, 1], ruta_final[i, 0]))

# Configurar los ejes
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar la gráfica
plt.show()

ruta_final_list = [tuple(coord) for coord in ruta_final.tolist()]

distancia_total = 0.0
for i in range(len(ruta_final_list) - 1):
    distancia = geodesic(ruta_final_list[i], ruta_final_list[i+1]).kilometers
    distancia_total += distancia

print("La distancia total recorrida es:", distancia_total, "kilómetros")


file = open("C:\\Users\\user\\Documents\\UiPath\\INTENTO\\ruta\\ruta.txt", "w")
for j in range(len(ruta)):
    file.write(ruta[j] + os.linesep)
    
    
file.close()






