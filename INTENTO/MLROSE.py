# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:53:31 2023

@author: user
"""

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
"Se cargan los datos dese el arcivo excel"
data = pd.read_excel("C:\\Users\\user\\Desktop\\Prueba.xlsx", header=None)

"Se organizan los datos para poder trabajar con ellos"
data = data.drop([1], axis=1)
datos=data.dropna()
datos= np.array(datos)
dim=np.shape(datos)

list_Coord=[]

info_Fin=[]
info=[]


# Función para convertir coordenadas geográficas a coordenadas cartesianas
def geo_to_cartesian(lat, lon):
    R = 6371 # Radio de la Tierra en km
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    return x, y

for i in range(dim[0]):
    txt="(?<=\/\@)[\d\.]+\,[\d\.\-]+"
    b=re.findall(txt,datos[i,8])
    b="".join(b)
    coord = list(map(float, b.split(',')))
    
    
    x, y = geo_to_cartesian(coord[0], coord[1]) # Convertir de geográficas a cartesianas
    list_Coord.append([x, y])
    info=datos[i,1]+" - "+datos[i,4]+" - "+datos[i,3]+"\n"+datos[i,7]
    info3=datos[i,4]+" - "+datos[i,5]
    info_Fin.append(info)
    

lista_Coord2 = np.array(list_Coord)

info_Fin=np.array(info_Fin)
list_Coord=np.array(list_Coord)

# Create fitness function object
fitness_coords = mlrose.TravellingSales(coords = lista_Coord2)

# Define the problem as a TSP
problem_fit = mlrose.TSPOpt(length = dim[0], fitness_fn = fitness_coords,
                            maximize=False)

# Define the problem as a TSP without fitness
problem_no_fit = mlrose.TSPOpt(length = dim[0], coords = lista_Coord2,
                               maximize=False)

# Set starting point as city 1 and run genetic algorithm
best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 0,
                                              pop_size=1000, mutation_prob=0.05,
                                              max_attempts=100 )

# Print the best route found
print('Mejor ruta ', best_state+1)
print('Distancia total recorrida: {:.2f} km'.format(best_fitness))

best_state_indices = np.concatenate(([best_state[-1]], best_state[:-1]))
best_state_coord = lista_Coord2[best_state_indices, :]
ruta_final=best_state_coord


valores =  range(1, len(ruta_final) + 1)  # enumeración del orden



lista = best_state
file = open("C:\\Users\\user\\Documents\\UiPath\\INTENTO\\ruta\\ruta.txt", "w")    
for j in range(len(lista)):
    valor=(str(lista[j]))
    file.write(str(info_Fin[int(valor)]) + os.linesep)
    
file.close()
 
# Graficar puntos y trazar la línea del recorrido
plt.scatter(ruta_final[:, 0], ruta_final[:, 1], c=valores, cmap='viridis')
plt.plot(ruta_final[:, 0], ruta_final[:, 1], '-o')

# Agregar etiquetas para los puntos
for i, valor in enumerate(valores):
    plt.annotate(valor, (ruta_final[i, 0], ruta_final[i, 1]))

# Configurar los ejes
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar la gráfica
plt.show()