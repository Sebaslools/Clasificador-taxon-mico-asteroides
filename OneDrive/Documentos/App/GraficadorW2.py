# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:36:47 2026

@author: SEBASTIAN
"""

import numpy as np #para las funciones
import matplotlib.pyplot as plt #Para las graficas
import pandas as pd #Para leer el archivo csv de SMASS2-E
import streamlit as st
import joblib
import requests
import io
from scipy.interpolate import interp1d #Para interpolar nuestros datos
from collections import Counter


eje_comun=np.linspace(0.44, 0.92, 49) #Para establecer el rango, incluyendo el limite superior y definiendo el numero de puntos 

@st.cache_data
def cargar_smassll():
    url_datos = "https://raw.githubusercontent.com/enriquebuendia/taxonomia_asteroides/refs/heads/main/data/tabla_smass.csv"
    df = pd.read_csv(url_datos)
    return df

@st.cache_resource
def cargar_PCA():
    url_pca = "https://github.com/enriquebuendia/taxonomia_asteroides/raw/refs/heads/main/data/PCA"
    pca_response = requests.get(url_pca)
    pca = joblib.load(io.BytesIO((pca_response.content)))
    return pca

@st.cache_resource
def cargar_NCA():
    url_nca = "https://github.com/enriquebuendia/taxonomia_asteroides/raw/refs/heads/main/data/NCA"
    nca_response = requests.get(url_nca)
    nca = joblib.load(io.BytesIO((nca_response.content)))
    return nca

@st.cache_resource
def cargar_KNN():
    url_KNN = "https://github.com/enriquebuendia/taxonomia_asteroides/raw/refs/heads/main/data/knnoriginal"
    KNN_response = requests.get(url_KNN)
    KNN = joblib.load(io.BytesIO((KNN_response.content)))
    url_KNNPCA = "https://github.com/enriquebuendia/taxonomia_asteroides/raw/refs/heads/main/data/knnPCA"
    KNNPCA_response = requests.get(url_KNNPCA)
    KNNPCA = joblib.load(io.BytesIO((KNNPCA_response.content)))
    url_KNNNCA = "https://github.com/enriquebuendia/taxonomia_asteroides/raw/refs/heads/main/data/knnNCA"
    KNNNCA_response = requests.get(url_KNNNCA)
    KNNNCA = joblib.load((io.BytesIO(KNNNCA_response.content)))
    return KNN, KNNPCA, KNNNCA

def leer_espectros(file, angstrom=False): #Para leer los datos del espectro
    data=np.loadtxt(file) #Para cargar el archivo txt
    long_de_onda=data[:, 0] #Tomando la info de la primera columna
    reflectancia=data[:, 1] #Tomando la info de la segunda columna
    
    
    if angstrom: #En caso de que el archivo este en angstroms convertimos a micras
        long_de_onda=long_de_onda/10000
        
    filtro= (long_de_onda>0.44) & (long_de_onda<0.92) #Definimos un filtro para los valores que tomaremos nuestros espectros
    
    if np.min(long_de_onda) > 0.44 or np.max(long_de_onda) < 0.92: #Para asegurarnos de que nuestros espectros esten en el rango esperado
        # Esto detiene el programa por completo
        raise ValueError(f"CRÍTICO: El espectro en '{file}' no cubre el rango necesario (0.44-0.92).")
                
    return long_de_onda[filtro], reflectancia[filtro] #Obtenemos la longitud de onda y la reflectancia de nuestros espectros pero solo en el rango que nos interesa gracias al filtro

def interpolación(long_de_onda, reflectancia): #Para interpolar los datos de nuestros espectros 
    f=interp1d(long_de_onda, reflectancia, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    return f(eje_comun) #Obtenemos los datos en los puntos que necesitamos para nuestro analisis (el eje comun)

def dist_euclidiana(espectro1, espectro2): #Para definir que tan diferentes son nuestros espectros de los del catalogo de SMASS2
    return np.sqrt(np.sum((espectro1 - espectro2)**2))

espectros = []
ks = []
catalogo = []


V_angstroms = ['Si', 'No']
m = list(range(1, 11))
km = np.arange(1, 101,1)

with st.spinner("Conectando con base de datos y cargando SMASSII..."):
    try: 
        df_smass = cargar_smassll()
        st.success(f"Catalogo SMASSII cargado")
        
    except Exception as e:
        st.error(f"No se pudo conectar con la base de datos: {e}")

espectro= st.file_uploader("Sube tus espectros txt", type="txt", accept_multiple_files=True)

with st.sidebar:
 ms = st.selectbox("¿Cuantas k se van a utilizar?", m)
 for i in range(ms):
   k = st.slider(f'k{i+1}', min_value= 10)
   ks.append(k)

left_column, middle_column, right_column = st.columns(3)

if left_column.checkbox("Todos los archivos en angstrom"):
    es_angstrom = True
    for file in espectro:
        try:
            l, r = leer_espectros(file, es_angstrom)
            interp = interpolación(l, r)
            espectros.append({"Nombre": file.name, "onda": l, "reflectancia": r, "interpolado": interp} )
        except ValueError as e:
            st.write("Error en archivo")
    
if middle_column.checkbox("Todos los archivos en micras"):
    es_angstrom = False
    for file in espectro:
        try:
            l, r = leer_espectros(file, es_angstrom)
            interp = interpolación(l, r)
            espectros.append({"Nombre": file.name, "onda": l, "reflectancia": r, "interpolado": interp} )
        except ValueError as e:
            st.write("Error en archivo")

if right_column.checkbox("Seleccionar manualmente unidades de cada archivo"):
 if espectro:
    st.write(f"Se clasificaran **{len(espectro)}** archivo(s)")
    for file in espectro:
        unidades = st.selectbox(f"'{file.name}' esta en angstroms?", V_angstroms)
        es_angstrom = True if unidades == 'Si' else False
        
        try:
            l, r = leer_espectros(file, es_angstrom)
            interp = interpolación(l, r)
            espectros.append({"Nombre": file.name, "onda": l, "reflectancia": r, "interpolado": interp} )
        except ValueError as e:
            st.write("Error en archivo")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Mis datos graficados", "Clasificación taxonomica (Euclidiana)", "PCA", "NCA", "KNN"])

with tab1:        
  if len(espectros)>0: #Para graficar nuestros espectros en la misma grafica
    fig, ax = plt.subplots()
    for item in espectros:
        ax.plot(item["onda"], item["reflectancia"], label=item["Nombre"])
    ax.set_xlabel("Longitud de onda (micras)")
    ax.set_ylabel("Reflectancia")
    ax.set_xlim(0.44, 0.92)
    ax.set_ylim(0.6, 1.4)
    ax.legend()
    ax.set_title("Espectros")
    st.pyplot(fig)
    plt.close()
  else:
    st.warning("No hay datos validos para graficar")

with tab2:        
  for i, fila in df_smass.iterrows():
    datos = fila.drop('tax')
    clase = fila['tax']
    catalogo.append({'clase': clase, 'datos': datos})
    

  for item in espectros:
   mis_datos_crudos = item["reflectancia"]
   mis_datos_ondas = item["onda"]
   mis_datos = item["interpolado"] #Aqui guardamos los valores que obtuvimos con la interpolacion
   distancias = [] #Aqui guardaremos la distancia entre nuestros espectros y los de SMASS
   nombres_clases = [] #Aqui guardamos el nombre de la clase a la que pertenece cada serie de datos
   datosc = []
   datostop = []
   for entrada in catalogo:
       d = dist_euclidiana(mis_datos, entrada["datos"]) #Obtenemos la separacion entre nuestros datos interpolados y los de SMASS
       datosc.append(entrada["datos"])
       distancias.append(d)
       nombres_clases.append(entrada["clase"]) #Guardamos el nombre de la clase al que pertenece esta distancia
   
   datosc = np.array(datosc)
   distancias = np.array(distancias) #Para mayor facilidad utilizamos array para trabajar los datos
   nombres_clases = np.array(nombres_clases) 
   idx = np.argsort(distancias) #Esto crea un indice ordenado de menor a mayor valor para las distancias
   
   if ms==0: 
    st.warning('Elija la cantidad de k a utilizar')
   else: 
    for valork in ks:
     if valork < 1:
         st.warning('Elija un valor de k valido')
     else:
      top = idx[:valork] #Creamos nuestro top dependiendo del valor de k que usemos
      puntajes = {} #Aqui guardaremos los puntajes para cada clase
      frecuencia = {} #Aqui guardaremos la frecuencia con la que aparece cada clase en el top
      peso = {} #Aqui guardamos el peso de cada clase
      suma_pesos = 0 #Aqui llevaremos la suma total de los pesos para la funcion de peso
      funcion_peso = {} #Aqui guardamos los valores obtenidos por la función de peso
      st.text(f"Coincidencias para {item['Nombre']}")
      for i in top:
         lista_clases = []
         nombres_clases1 = nombres_clases[i]
         lista_clases.append(nombres_clases1)
         datosc1 = datosc[i]
         datostop.append({'clase': nombres_clases1,'onda':eje_comun ,'datos': datosc1})
         st.text(f"{nombres_clases[i]}: d = {distancias[i]}") #Aqui mostramos las clases y distancias en el top k
      for i, indice in enumerate(top):
         clase = nombres_clases[indice] #Aqui obtenemos el nombre de la clase usando el indice en top
         puntos = valork - i #Aqui calculamos el puntaje a cada clase
         puntajes[clase] = puntajes.get(clase,0) + puntos #Aqui vamos guardando los puntajes para cada clase y sumandolos
         frecuencia[clase] = frecuencia.get(clase, 0) + 1 #Aqui calculamos la frecuencia con la que aparece cada clase y la guardamos
      for clase in puntajes:
         if clase in frecuencia: 
          peso[clase] = puntajes[clase] * frecuencia[clase]/valork #Aqui calculamos los pesos de cada clase
          suma_pesos += peso[clase] #Calculamos el valor total de los pesos y lo guardamos
      st.text("Peso de cada clase")
      for clase in peso:
         funcion_peso[clase] = (peso[clase]/suma_pesos)*100 #Calculamos la función de peso para cada clase
         st.text(f"{clase}: {funcion_peso[clase]:.2f}%") #Mostramos el valor de la función de peso 
      if len(datostop)>0: #Para graficar nuestros espectros en la misma grafica
          clases_top = Counter(item['clase'] for item in datostop)
          clases_unicas = [clase for clase, freq in clases_top.most_common()]
          clases_seleccionadas = st.multiselect("Clases a mostrar:", options=clases_unicas, default=clases_unicas, key="filtro clase graficas" )
          datos_filtrados = [item for item in datostop if item['clase'] in clases_seleccionadas]
          if len(datos_filtrados)>0:
           cmap = plt.get_cmap('tab10')
           colores_clases = {}
           for i, clase in enumerate(clases_unicas):
              colores_clases[clase] = cmap(i%10)
           fig, ax = plt.subplots()
           clases_lista = set()
           for item in datos_filtrados:
              clase_actual = item['clase']
              color_actual = colores_clases[clase_actual]
              if item["clase"] not in clases_lista:
                  etiqueta = f"Clase {item['clase']}"
                  clases_lista.add(item['clase'])
              else:
                  etiqueta = "_nolegend_"
              ax.plot(item["onda"], item["datos"], color = color_actual ,label=f"{etiqueta} ({frecuencia[item['clase']]})", alpha = 0.5)
           ax.plot(mis_datos_ondas, mis_datos_crudos, color = 'black', linestyle='--', label = "Mi espectro", alpha = 0.5)
           ax.set_xlabel("Longitud de onda (micras)")
           ax.set_ylabel("Reflectancia")
           ax.set_xlim(0.44, 0.92)
           ax.set_ylim(0.6, 1.4)
           ax.legend()
           ax.set_title("Espectros")
           st.pyplot(fig)
           plt.close()
          else:
           st.warning("No hay datos validos para graficar")
           
    with tab3:
    pca = cargar_PCA()
    for item in espectros:
        mis_datos = item["interpolado"]
        mis_datos2d = np.array(mis_datos).reshape(1, -1)
        mis_datosPCA = pca.transform(mis_datos2d)
        prediccionPCA = knnpca.predict_proba(mis_datosPCA)
        indice_clase = np.argmax(prediccionPCA)
        clase_predicha = knnpca.classes_[indice_clase]
        st.text(f"Para {item['Nombre']} la clase más probable es {clase_predicha} de acuerdo con el modelo KNN-PCA")
        clases = knnpca.classes_
        probabilidades = knnpca.predict_proba(mis_datosPCA)[0]
        clases_y_probas = sorted(zip(clases, probabilidades), key=lambda x: x[1], reverse = True)
        for clase, proba in clases_y_probas:
            porcentaje = proba*100
            if porcentaje > 0.0:
                st.write(f'**Clase {clase}**: {porcentaje:.2f}%')
with tab4:
    nca = cargar_NCA()
    for item in espectros:
        mis_datos = item["interpolado"]
        mis_datos2d = np.array(mis_datos).reshape(1,-1)
        mis_datosNCA = nca.transform(mis_datos2d)
        prediccionNCA = knnnca.predict_proba(mis_datosNCA)
        indice_clase = np.argmax(prediccionNCA)
        clase_predicha = knnnca.classes_[indice_clase]
        st.text(f"Para {item['Nombre']} la clase más probable es {clase_predicha} de acuerdo con el modelo KNN-NCA")
        clases = knnnca.classes_
        probabilidades = knnnca.predict_proba(mis_datosNCA)[0]
        clases_y_probas = sorted(zip(clases, probabilidades), key=lambda x: x[1], reverse = True)
        for clase, proba in clases_y_probas:
            porcentaje = proba*100
            if porcentaje > 0.0:
                st.write(f'**Clase {clase}**: {porcentaje:.2f}%')
        
with tab5:
    for item in espectros:
        mis_datos = item["interpolado"]
        mis_datos2d = np.array(mis_datos).reshape(1,-1)
        prediccionKNN = knn.predict_proba(mis_datos2d)
        indice_clase = np.argmax(prediccionKNN)
        clase_predicha = knn.classes_[indice_clase]
        st.text(f"Para {item['Nombre']} la clase más probable es {clase_predicha} de acuerdo al modelo KNN")
        clases = knn.classes_
        probabilidades = knn.predict_proba(mis_datos2d)[0]
        clases_y_probas = sorted(zip(clases, probabilidades), key=lambda x: x[1], reverse = True)
        for clase, proba in clases_y_probas:
            porcentaje = proba*100
            if porcentaje > 0.0:
                st.write(f'**Clase {clase}**: {porcentaje:.2f}%')
    
    
