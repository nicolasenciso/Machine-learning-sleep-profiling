import csv, operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics
import random
import math
import os


class SinglePoint:
    def __init__(self,vectorPersona):
        self.vector = vectorPersona
        self.cluster = "n"
        self.distancia = 0

class DBscan:
    def __init__(self, points):#Point es una lista de objetos point que son los que tienen los datos de cada persona
        self.epsilon = 0
        self.minPts = 0

    def seleccionPunto(self,points):#selecciona el punto inicial para medicion de distancias
        punto = random.randint(1,len(points))
        inicialPunto = points[punto]
        return inicialPunto

    def medicionDistancias(self, inicialPunto, points):
        for punto in points:
            euclidiana = []
            edad = int(punto.vector[0])
            edadInit = int(inicialPunto.vector[0])
            distEdad = (edad - edadInit)**2
            euclidiana.append(distEdad)

            dormir = int(punto.vector[1])
            dormirInit = int(inicialPunto.vector[1])
            distDormir = (dormir - dormirInit)**2
            euclidiana.append(distDormir)

            animo = int(punto.vector[2])
            animoInit = int(inicialPunto.vector[2])
            distAnimo = (animo - animoInit)**2
            euclidiana.append(distAnimo)

            gustosList = []
            for i in range(len(punto.vector[3])):
                gusto = int(punto.vector[3][i])
                gustoInit = int(inicialPunto.vector[3][i])
                distGusto = (gusto - gustoInit)**2
                gustosList.append(distGusto)
            finalGustos = sum(gustosList)
            euclidiana.append(finalGustos)

            productividad = int(punto.vector[4])
            productividadInit = int(inicialPunto.vector[4])
            distProductividad = (productividad - productividadInit)**2
            euclidiana.append(distProductividad)

            genero = int(punto.vector[5])
            generoInit = int(inicialPunto.vector[5])
            distGenero = (genero - generoInit)**2
            euclidiana.append(distGenero)

            ocupacion = int(punto.vector[6])
            ocupacionInit = int(inicialPunto.vector[6])
            distOcupacion = (ocupacion - ocupacionInit)**2
            euclidiana.append(distOcupacion)

            fisico = int(punto.vector[7])
            fisicoInit = int(inicialPunto.vector[7])
            distFisico = (fisico - fisicoInit)**2
            euclidiana.append(distFisico)

            distanciaFinal = math.sqrt(sum(euclidiana))
            punto.distancia = distanciaFinal

    def dbscann(self, xvector, yvector,epsilon, minPts):
        puntos = []
        for i in range(len(xvector)):
            puntos.append([xvector[i] ,yvector[i]])
        puntos = StandardScaler().fit_transform(puntos)
        db = DBSCAN(eps=epsilon, min_samples=minPts).fit(puntos) #SELECCION EPSILON Y GRUPO VECINOS
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print (labels)#clasificacion de los puntos segun el color y con mismo orden de punto
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #print(n_clusters_)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = puntos[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
            xy = puntos[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
        plt.title('Numero estimado de clusters: %d' % n_clusters_)
        plt.axis([-2.5,5, -2.5,5])
        #plt.show()
        return labels
    
    def paridad(self, points, par1, epsilon, minPts, puntoEstudio):
        self.epsilon = epsilon
        self.minPts = minPts
        x = int(puntoEstudio[par1[0]])
        y = int(puntoEstudio[par1[1]])
        xvector, yvector = [], []
        xvector.append(x)
        yvector.append(y)
        for punto in points:
            xvector.append(punto.vector[par1[0]])
            yvector.append(punto.vector[par1[1]])
        
        salida = self.dbscann(xvector, yvector,epsilon,minPts)
        return salida

    def kmeans(self, points, punto, k):
        salidadCompletaK = []
        edadk, dormirk, productividadk = 0,0,0
        muyFelizk, felizk, normalk, tristek, muyTristek = 0,0,0,0,0
        animok = [0,0,0,0,0]
        hombrek, mujerk, otrok = 0,0,0
        generok = [0,0,0]
        estudiantek, trabajadork, sin_ocupacionk, eytk, eysk, tysk,etsk = 0,0,0,0,0,0,0
        ocupacionk = [0,0,0,0,0,0,0]
        somnolenciak, resacak, enfermedadk, ejerciciok, normalFisicok = 0,0,0,0,0
        fisicok = [0,0,0,0,0]
        self.medicionDistancias(punto,points)
        distancias = []
        for caso in points:
            distancias.append(caso.distancia)
        distancias.sort()
        vecinos = distancias[0:k]
        perfiles = []
        for veci in vecinos:
            for caso in points:
                if caso.distancia == veci:
                    if veci not in perfiles:
                        perfiles.append(caso)
        perfiles = perfiles[0:k]
        aux = []
        for perfil in perfiles:
            for i in range(8):
                if i == 0:
                    edadk = edadk + int(perfil.vector[i])
                    aux.append(edadk)
                elif i == 1:
                    dormirk = dormirk + int(perfil.vector[i])
                elif i == 2:
                    if float(perfil.vector[i]) == 1:
                        muyFelizk += 1
                    elif float(perfil.vector[i]) == 0.8:
                        felizk += 1
                    elif float(perfil.vector[i]) == 0.6:
                        normalk += 1
                    elif float(perfil.vector[i]) == 0.4:
                        tristek += 1
                    elif float(perfil.vector[i]) == 0.2:
                        muyTristek += 1
                elif i == 4:
                    productividadk = productividadk + int(perfil.vector[i])
                elif i == 5:
                    if float(perfil.vector[i]) == 0.33:
                        mujerk += 1
                    elif float(perfil.vector[i]) == 0.66:
                        hombrek += 1
                    else:
                        otrok += 1
                elif i == 6:
                    if float(perfil.vector[i]) == 0.14:
                        estudiantek += 1
                    elif float(perfil.vector[i]) == 0.28:
                        trabajadork += 1
                    elif float(perfil.vector[i]) == 0.42:
                        sin_ocupacionk += 1
                    elif float(perfil.vector[i]) == 0.56:
                        eytk += 1
                    elif float(perfil.vector[i]) == 0.7:
                        eysk += 1
                    elif float(perfil.vector[i]) == 0.84:
                        tysk += 1
                    else:
                        etsk += 1
                elif i == 7:
                    if float(perfil.vector[i]) == 0.2:
                        somnolenciak += 1
                    elif float(perfil.vector[i]) == 0.4:
                        resacak += 1
                    elif float(perfil.vector[i]) == 0.6:
                        enfermedadk += 1
                    elif float(perfil.vector[i]) == 0.8:
                        ejerciciok += 1
                    else:
                        normalFisicok += 1
        #edad,dormir,animo,gustos,productividad,genero,ocupacion,fisico
        edadk = edadk/len(aux)
        salidadCompletaK.append(edadk)

        dormirk = dormirk/len(aux)
        salidadCompletaK.append(dormirk)

        animok = [muyFelizk, felizk, normalk, tristek, muyTristek]
        salidadCompletaK.append(animok)

        productividadk = productividadk/len(aux)
        salidadCompletaK.append(productividadk)

        generok = [hombrek, mujerk, otrok]
        salidadCompletaK.append(generok)

        ocupacionk = [estudiantek, trabajadork, sin_ocupacionk, eytk, eysk, tysk,etsk]
        salidadCompletaK.append(ocupacionk)

        fisicok = [somnolenciak, resacak, enfermedadk, ejerciciok, normalFisicok]
        salidadCompletaK.append(fisicok)
        
        #print(salidadCompletaK)
        return salidadCompletaK


      

def definicionPuntos():
    datos = open('form.csv','rb')
    persona = []
    while csv.reader(datos):
        try:
            respuesta = []
            entrada = csv.reader(datos)
            fecha,edad,dormir,animo,gustos,productividad,genero,ocupacion,fisico = next(entrada)
            respuesta.append(edad)
            respuesta.append(dormir)
            aanimo = 0
            if animo == 'Muy feliz':
                aanimo = 1
            elif animo == 'Feliz':
                aanimo = 0.8
            elif animo == 'Normal':
                aanimo = 0.6
            elif animo == 'Triste':
                aanimo = 0.4
            else:
                aanimo = 0.2
            respuesta.append(aanimo)
        
            ggustos = [] 
            if 'Cine' in gustos or ' Cine' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0) 
            if 'Literatura' in gustos or ' Literatura' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Musica' in gustos or ' Musica' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Deportes' in gustos or ' Deportes' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Arte' in gustos or ' Arte' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Tecnologia' in gustos or ' Tecnologia' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Ciencias exactas' in gustos or ' Ciencias exactas' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Gastronomia' in gustos or ' Gastronomia' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Manualidades' in gustos or ' Manualidades' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            if 'Ciencias sociales y humanas' in gustos or ' Ciencias sociales y humanas' in gustos: 
                ggustos.append(1) 
            else: ggustos.append(0)
            respuesta.append(ggustos)
            
            respuesta.append(productividad)
            
            ggenero = 0
            if genero == 'Mujer':
                ggenero = 0.33
            elif genero == 'Hombre':
                ggenero = 0.66
            else:
                ggenero = 1
            respuesta.append(ggenero)

            oocupacion = 0
            if ocupacion == 'Estudiante':
                oocupacion = 0.14
            elif ocupacion == 'Trabajador':
                oocupacion = (0.28)
            elif ocupacion == 'Sin ocupacion':
                oocupacion = (0.42)
            elif 'Estudiante' in ocupacion and 'Trabajador' in ocupacion:
                oocupacion = (0.56)
            elif 'Estudiante' in ocupacion and 'Sin ocupacion' in ocupacion:
                oocupacion = (0.7)
            elif 'Trabajador' in ocupacion and 'Sin ocupacion' in ocupacion:
                oocupacion = (0.84)
            else:
                oocupacion = (1)
            respuesta.append(oocupacion)
            
            ffisico = 0
            if fisico == 'Somnolencia':
                ffisico = 0.2
            elif fisico == 'Resaca':
                ffisico = 0.4
            elif fisico == 'Enfermedad':
                ffisico = 0.6
            elif fisico == 'Ejercicio':
                ffisico = 0.8
            else:
                ffisico = 1
            respuesta.append(ffisico)
            persona.append(respuesta)
            #print respuesta
        except StopIteration:
            break
    return persona

def objetizacionPersonas(persona):
    points = []
    for persona in personas:
        auxPersona = SinglePoint(persona)
        points.append(auxPersona)
    return points

def grafiaciones(pp,scan, epsilon, minPts, puntoEstudio):
    sal = []
    for i in range(7):
        for j in range(7):
            if (i != 3 and j != 3) and (i != j):
                ss = scan.paridad(pp,(i,j),epsilon, minPts, puntoEstudio)
                sal.append(ss)
    return sal

def perfilamientoClustering(points, salida):
    total = []
    salidaCompleta = []
    edad, dormir, productividad = 0,0,0
    muyFeliz, feliz, normal, triste, muyTriste = 0,0,0,0,0
    animo = [0,0,0,0,0]
    hombre, mujer, otro = 0,0,0
    genero = [0,0,0]
    estudiante, trabajador, sin_ocupacion, eyt, eys, tys,ets = 0,0,0,0,0,0,0
    ocupacion = [0,0,0,0,0,0,0]
    somnolencia, resaca, enfermedad, ejercicio, normalFisico = 0,0,0,0,0
    fisico = [0,0,0,0,0]
    for clusterado in salida:
        posSimilares = []
        actual = clusterado[0]
        for i in range(len(clusterado)-1):
            if clusterado[i] == actual:
                posSimilares.append(i)
        total.append(posSimilares)
    perfiles = []
    aux = []
    for caso in total:
        for elem in caso:
            perfiles.append(points[elem])
        for perfil in perfiles:
            for i in range(8):
                if i == 0:
                    edad = edad + int(perfil.vector[i])
                    aux.append(edad)
                elif i == 1:
                    dormir = dormir + int(perfil.vector[i])
                elif i == 2:
                    if float(perfil.vector[i]) == 1:
                        muyFeliz += 1
                    elif float(perfil.vector[i]) == 0.8:
                        feliz += 1
                    elif float(perfil.vector[i]) == 0.6:
                        normal += 1
                    elif float(perfil.vector[i]) == 0.4:
                        triste += 1
                    elif float(perfil.vector[i]) == 0.2:
                        muyTriste += 1
                elif i == 4:
                    productividad = productividad + int(perfil.vector[i])
                elif i == 5:
                    if float(perfil.vector[i]) == 0.33:
                        mujer += 1
                    elif float(perfil.vector[i]) == 0.66:
                        hombre += 1
                    else:
                        otro += 1
                elif i == 6:
                    if float(perfil.vector[i]) == 0.14:
                        estudiante += 1
                    elif float(perfil.vector[i]) == 0.28:
                        trabajador += 1
                    elif float(perfil.vector[i]) == 0.42:
                        sin_ocupacion += 1
                    elif float(perfil.vector[i]) == 0.56:
                        eyt += 1
                    elif float(perfil.vector[i]) == 0.7:
                        eys += 1
                    elif float(perfil.vector[i]) == 0.84:
                        tys += 1
                    else:
                        ets += 1
                elif i == 7:
                    if float(perfil.vector[i]) == 0.2:
                        somnolencia += 1
                    elif float(perfil.vector[i]) == 0.4:
                        resaca += 1
                    elif float(perfil.vector[i]) == 0.6:
                        enfermedad += 1
                    elif float(perfil.vector[i]) == 0.8:
                        ejercicio += 1
                    else:
                        normalFisico += 1

    #edad,dormir,animo,gustos,productividad,genero,ocupacion,fisico
    edad = edad/len(aux)
    salidaCompleta.append(edad)

    dormir = dormir/len(aux)
    salidaCompleta.append(dormir)

    animo = [muyFeliz, feliz, normal, triste, muyTriste]
    salidaCompleta.append(animo)
    
    productividad = productividad/len(aux)
    salidaCompleta.append(productividad)

    genero = [hombre, mujer, otro]
    salidaCompleta.append(genero)

    ocupacion = [estudiante, trabajador, sin_ocupacion, eyt, eys, tys,ets]
    salidaCompleta.append(ocupacion)

    fisico = [somnolencia, resaca, enfermedad, ejercicio, normalFisico]
    salidaCompleta.append(fisico)

    #print(salidaCompleta)
    return salidaCompleta
    
    
    
    
                
def perfilamientoTotalFinal(salidadCompletaK,salidaCompleta):
    final = []
    for i in range(len(salidaCompleta)):
        if i == 2:
            aux = []
            for j in range(5):
                aux.append((salidaCompleta[i][j] + salidadCompletaK[i][j])/2)
            final.append(aux)
        elif i == 4:
            aux = []
            for j in range(3):
                aux.append((salidaCompleta[i][j] + salidadCompletaK[i][j])/2)
            final.append(aux)
        elif i == 5:
            aux = []
            for j in range(7):
                aux.append((salidaCompleta[i][j] + salidadCompletaK[i][j])/2)
            final.append(aux)
        elif i == 6:
            aux = []
            for j in range(5):
                aux.append((salidaCompleta[i][j] + salidadCompletaK[i][j])/2)
            final.append(aux)
        else:
            final.append((salidaCompleta[i] + salidadCompletaK[i])/2)
    

    #edad,dormir,animo,gustos,productividad,genero,ocupacion,fisico
    animo = final[2]
    maxAnimo = max(animo)
    indiAnimo = animo.index(maxAnimo)
    animoFinal = ""
    if indiAnimo == 0: animoFinal = "muyFeliz"
    if indiAnimo == 1: animoFinal = "feliz"
    if indiAnimo == 2: animoFinal = "normal"
    if indiAnimo == 3: animoFinal = "triste"
    if indiAnimo == 4: animoFinal = "muyTriste"

    genero = final[4]
    maxGen = max(genero)
    maximo = genero[0]
    indiGen = 0
    for i in range(len(genero)):
        if genero[i] > maximo:
            maximo = genero[i]
            indiGen = i
        
    genFinal = ""
    if indiGen == 0: genFinal = "hombre"
    if indiGen == 1: genFinal = "mujer"
    if indiGen == 2: genFinal = "otro genero"

    ocupacion = final[5]
    maxOcu = max(ocupacion)
    indiOcu = ocupacion.index(maxOcu)
    ocuFinal = ""
    if indiOcu == 0: ocuFinal = "estudiante"
    if indiOcu == 1: ocuFinal = "trabajador"
    if indiOcu == 2: ocuFinal = "sin ocupacion"
    if indiOcu == 3: ocuFinal = "estudiante y trabajador"
    if indiOcu == 4: ocuFinal = "estudiante y sin ocupacion"
    if indiOcu == 5: ocuFinal = "trabajador y sin ocupacion"
    if indiOcu == 6: ocuFinal = "estudiante, trabajador y sin ocupacion"

    fisico = final[6]
    maxFis = max(fisico)
    indiFis = fisico.index(maxFis)
    fisFinal = ""
    if indiFis == 0: fisFinal = "somnolencia"
    if indiFis == 1: fisFinal = "resaca"
    if indiFis == 2: fisFinal = "enfermedad"
    if indiFis == 3: fisFinal = "ejercicio"
    if indiFis == 4: fisFinal = "normal"
    os.system('clear')
    print("")
    print("----------------------RESULTADOS-------------------------")
    print("")
    print("Evaluacion perfilamiento de persona entrante: ")
    print("")
    print("Analisis hecho por medio de clustering por pares de variables y k vecinos en 17 dimensiones")
    print("")
    print("Edad: " + str(final[0]))
    print("")
    print("Estado de animo:  "+animoFinal)
    print("")
    print("Posible genero: "+genFinal)
    print("")
    print("Posible ocupacion:  "+ocuFinal)
    print("")
    print("Posible estado fisico de ultima semana:  "+fisFinal)
    print("")
    print("-----------------------------------------------------------")
    print("")
    print("")
    print("-----------------------------------------------------------")
    print("")
    print("Recomendaciones de horas de suenio: ")
    print("")
    print("De acuerdo a nuestro analisis, le recomendamos dormir un minimo de:  "+str(final[1])+"  horas de descanso")
    print("")
    print("Con jornadas de trabajo sumado total de:  "+str(final[3])+ " horas de trabajo totalizado")
    print("")
    print("")
    print("-------------------------------------------------------------")
    print("")
    print("Se recomienda tener 17 minutos de descanso por cada 52 minutos de trabajo")
    print("")
    print("-----------------------------------------------------------------")
    return (final[1],final[3])

def entradaNuevoDato():
    datos = open('new.csv','rb')
    persona = []
    try:
        respuesta = []
        entrada = csv.reader(datos)
        fecha,edad,dormir,animo,gustos,productividad,genero,ocupacion,fisico = next(entrada)
        respuesta.append(edad)
        respuesta.append(dormir)
        aanimo = 0
        if animo == 'Muy feliz':
            aanimo = 1
        elif animo == 'Feliz':
            aanimo = 0.8
        elif animo == 'Normal':
            aanimo = 0.6
        elif animo == 'Triste':
            aanimo = 0.4
        else:
            aanimo = 0.2
        respuesta.append(aanimo)
        
        ggustos = [] 
        if 'Cine' in gustos or ' Cine' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0) 
        if 'Literatura' in gustos or ' Literatura' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Musica' in gustos or ' Musica' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Deportes' in gustos or ' Deportes' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Arte' in gustos or ' Arte' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Tecnologia' in gustos or ' Tecnologia' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Ciencias exactas' in gustos or ' Ciencias exactas' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Gastronomia' in gustos or ' Gastronomia' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Manualidades' in gustos or ' Manualidades' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        if 'Ciencias sociales y humanas' in gustos or ' Ciencias sociales y humanas' in gustos: 
            ggustos.append(1) 
        else: ggustos.append(0)
        respuesta.append(ggustos)
            
        respuesta.append(productividad)
            
        ggenero = 0
        if genero == 'Mujer':
            ggenero = 0.33
        elif genero == 'Hombre':
            ggenero = 0.66
        else:
            ggenero = 1
        respuesta.append(ggenero)

        oocupacion = 0
        if ocupacion == 'Estudiante':
            oocupacion = 0.14
        elif ocupacion == 'Trabajador':
            oocupacion = (0.28)
        elif ocupacion == 'Sin ocupacion':
            oocupacion = (0.42)
        elif 'Estudiante' in ocupacion and 'Trabajador' in ocupacion:
            oocupacion = (0.56)
        elif 'Estudiante' in ocupacion and 'Sin ocupacion' in ocupacion:
            oocupacion = (0.7)
        elif 'Trabajador' in ocupacion and 'Sin ocupacion' in ocupacion:
            oocupacion = (0.84)
        else:
            oocupacion = (1)
        respuesta.append(oocupacion)
            
        ffisico = 0
        if fisico == 'Somnolencia':
            ffisico = 0.2
        elif fisico == 'Resaca':
            ffisico = 0.4
        elif fisico == 'Enfermedad':
            ffisico = 0.6
        elif fisico == 'Ejercicio':
            ffisico = 0.8
        else:
            ffisico = 1
        respuesta.append(ffisico)
        persona.append(respuesta)
        #print respuesta
    except StopIteration:
        print("error EOF")

    return persona[0]

puntoEstudio = entradaNuevoDato()
personas = definicionPuntos()
pp = objetizacionPersonas(personas)
scan = DBscan(pp)
scan.epsilon = 0.6
scan.minPts = 4
punto = scan.seleccionPunto(pp)
scan.medicionDistancias(punto,pp)
salk = scan.kmeans(pp,pp[4],7)
resultado = grafiaciones(pp,scan,scan.epsilon,scan.minPts, puntoEstudio)
salC = perfilamientoClustering(pp,resultado)
dormir, trabajar = perfilamientoTotalFinal(salk,salC )
parHoras = (dormir,trabajar)
f = open("horas.txt",'w')
f.write(str(parHoras))
f.close()

#La entrada de datos es un archivo csv que se llama new.csv con una unica fila
#se hace k means y clustering de a pares de variables
#La graficacion de los clusters esta comentada en dbscann() en plot.show()
#la salida de horas de (descanso,trabajo) en el archivo horas.txt

#puntoEstudio = ['22', '5', 0.6, [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], '8', 0.33, 0.14, 0.2]
#analisis = ejecucion(puntoEstudio)