import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import cv2
import tempfile
from pathlib import Path
import os
from evolucion import EvolucionPoblacion
from evolucion.funcion import function_objetivo
from video_utils.utils import save_frame, generate_video
from gui_setup import setup_gui

def ejecutar_algoritmo():
    for widget in frame_grafica.winfo_children():
        widget.destroy()

    # Configuración inicial
    minimizar = False
    evolucion = EvolucionPoblacion.EvolucionPoblacion(minimizar=minimizar)

    try:
        evolucion.temp_dir = Path(tempfile.mkdtemp()) # aquí modificamos
        os.makedirs(evolucion.temp_dir, exist_ok=True)
        print(f"Created temporary directory: {evolucion.temp_dir}")
    except Exception as e:
        print(f"Error creating temporary directory: {e}")
        return
        
    intervalo = (float(entry_intervalo_inicio.get()), float(entry_intervalo_fin.get()))
    resolucion = float(entry_resolucion.get())
    cantidad_poblacion = int(entry_cantidad_poblacion.get())
    probabilidad_cruza = float(entry_probabilidad_cruza.get())
    probabilidad_mutacion = float(entry_probabilidad_mutacion.get())
    probabilidad_mutacion_bits = float(entry_probabilidad_mutacion_bits.get())
    num_generaciones = int(entry_num_generaciones.get())

    # === Inicialización (parámetrización) ===
    A, B = intervalo
    n_original = int(np.ceil((B - A) / resolucion)) + 1
    num_bits = int(np.ceil(np.log2(n_original)))
    delta_x_star = (B - A) / (2**num_bits - 1)
    indices = np.arange(2**num_bits)
    valores_x = A + indices * delta_x_star
    valores_x_filtrados = valores_x[valores_x <= B]
    representaciones_binarias = [format(i, f'0{num_bits}b') for i in range(len(valores_x_filtrados))]
    
    # Configuración de la figura
    evolucion.fig = Figure(figsize=(12, 8))
    evolucion.ax1 = evolucion.fig.add_subplot(311)  # Función objetivo y población actual
    evolucion.ax2 = evolucion.fig.add_subplot(312)  # Evolución del fitness
    evolucion.ax3 = evolucion.fig.add_subplot(313)  # Población en generación actual
    
    # Configurar gráficas iniciales
    x_continuo = np.linspace(A, B, 1000)
    y_continuo = function_objetivo(x_continuo)
    evolucion.ax1.plot(x_continuo, y_continuo, 'b-', label='f(x)')
    evolucion.ax1.set_title('Función Objetivo y Población Actual')
    evolucion.ax1.grid(True)
    evolucion.ax1.legend()

    evolucion.ax2.set_title('Evolución de la Aptitud')
    evolucion.ax2.set_xlabel('Generaciones')
    evolucion.ax2.set_ylabel('Aptitud')
    evolucion.ax2.grid(True)

    evolucion.ax3.set_title('Población en Generación Actual')
    evolucion.ax3.set_xlabel('x')
    evolucion.ax3.set_ylabel('f(x)')
    evolucion.ax3.grid(True)
    
    def init():
        evolucion.poblacion = np.random.choice(len(valores_x_filtrados), size=cantidad_poblacion, replace=True)
        return []

    def animate(frame):
        # Limpiar gráficas anteriores
        for collection in evolucion.ax1.collections:
            collection.remove()
        for collection in evolucion.ax3.collections:
            collection.remove()
        
        # Evaluar aptitudes
        x_poblacion = valores_x_filtrados[evolucion.poblacion]
        aptitudes = function_objetivo(x_poblacion)

        if evolucion.minimizar:
            indices_ordenados = np.argsort(aptitudes)
        else:
            indices_ordenados = np.argsort(aptitudes)[::-1] # Orden ascendente para la máximización
        
        # Actualizar población actual en la primera gráfica
        scatter = evolucion.ax1.scatter(x_poblacion, aptitudes, c='gray', alpha=0.5, label='Población')
        mejor_x = x_poblacion[indices_ordenados[0]]
        mejor_y = aptitudes[indices_ordenados[0]]
        peor_x = x_poblacion[indices_ordenados[-1]]
        peor_y = aptitudes[indices_ordenados[-1]]
        
        evolucion.ax1.scatter([mejor_x], [mejor_y], c='g', s=100, label='Mejor')
        evolucion.ax1.scatter([peor_x], [peor_y], c='r', s=100, label='Peor')
        
        # Actualizar gráfica de evolución
        evolucion.mejores.append(mejor_y)
        evolucion.peores.append(peor_y)
        evolucion.promedios.append(np.mean(aptitudes))
        
        generaciones = np.arange(len(evolucion.mejores))
        evolucion.ax2.clear()
        evolucion.ax2.plot(generaciones, evolucion.mejores, 'g-', label='Mejor')
        evolucion.ax2.plot(generaciones, evolucion.promedios, 'b-', label='Promedio')
        evolucion.ax2.plot(generaciones, evolucion.peores, 'r-', label='Peor')
        evolucion.ax2.set_title('Evolución de la Aptitud')
        evolucion.ax2.set_xlabel('Generaciones')
        evolucion.ax2.set_ylabel('Aptitud')
        evolucion.ax2.grid(True)
        evolucion.ax2.legend()
        
        # Actualizar gráfica de población actual
        evolucion.ax3.clear()
        evolucion.ax3.plot(x_continuo, y_continuo, 'b-', alpha=0.3)
        evolucion.ax3.scatter(x_poblacion, aptitudes, c='gray', alpha=0.5)
        evolucion.ax3.scatter([mejor_x], [mejor_y], c='g', s=100, label='Mejor')
        evolucion.ax3.scatter([peor_x], [peor_y], c='r', s=100, label='Peor')
        evolucion.ax3.set_title(f'Población en Generación {frame}')
        evolucion.ax3.set_xlabel('x')
        evolucion.ax3.set_ylabel('f(x)')
        evolucion.ax3.grid(True)
        evolucion.ax3.legend()

        # Actualizar resultados en la última generación
        if frame == num_generaciones - 1:
            mejor_indice = evolucion.poblacion[indices_ordenados[0]]
            evolucion.mejor_x_final = valores_x_filtrados[mejor_indice]
            evolucion.mejor_fx_final = function_objetivo(evolucion.mejor_x_final)
            evolucion.mejor_binario_final = representaciones_binarias[mejor_indice]
            evolucion.mejor_decimal_final = mejor_indice
            
            # Actualizar variables de resultados
            resultado_vars["cantidad_puntos"].set(str(n_original))
            resultado_vars["bits"].set(str(num_bits))
            resultado_vars["delta_x"].set(f"{delta_x_star:.6f}")
            resultado_vars["mejor_x"].set(f"{evolucion.mejor_x_final:.6f}")
            resultado_vars["mejor_fx"].set(f"{evolucion.mejor_fx_final:.6f}")
            resultado_vars["mejor_binario"].set(evolucion.mejor_binario_final)
            resultado_vars["mejor_decimal"].set(str(evolucion.mejor_decimal_final))
        
        if frame < num_generaciones - 1:
            parejas = []
            for i in range(len(evolucion.poblacion)):
                if np.random.rand() <= probabilidad_cruza:
                    j = np.random.randint(0, i + 1)
                    parejas.append((evolucion.poblacion[indices_ordenados[i]], 
                                evolucion.poblacion[indices_ordenados[j]]))
            
            # Generar descendientes
            descendientes = []
            for padre1, padre2 in parejas:
                bin_padre1 = representaciones_binarias[padre1]
                bin_padre2 = representaciones_binarias[padre2]
                punto_cruza = np.random.randint(1, num_bits - 1)
                hijo1 = bin_padre1[:punto_cruza] + bin_padre2[punto_cruza:]
                hijo2 = bin_padre2[:punto_cruza] + bin_padre1[punto_cruza:]
                descendientes.extend([int(hijo1, 2), int(hijo2, 2)])
            
            # ===== Mutación ====
            for i in range(len(descendientes)):
                if np.random.rand() <= probabilidad_mutacion:
                    bin_hijo = representaciones_binarias[descendientes[i]]
                    for bit in range(num_bits):
                        if np.random.rand() <= probabilidad_mutacion_bits:
                            bin_hijo = (bin_hijo[:bit] + 
                                    ('1' if bin_hijo[bit] == '0' else '0') + 
                                    bin_hijo[bit + 1:])
                    descendientes[i] = int(bin_hijo, 2)

            # ==== IMPLEMENTACIÓN DE LA PODA ====

            # 1. Combinar población actual con descendientes
            poblacion_combinada = np.concatenate([evolucion.poblacion, descendientes])
            
            # 2. Calcular aptitud de cada individuo
            x_combinada = valores_x_filtrados[poblacion_combinada]
            aptitudes_combinadas = function_objetivo(x_combinada)
            
            # 3. Mantener un solo ejemplar (eliminar repetidos)
            # Crear un array de índices únicos basados en los valores x
            _, indices_unicos = np.unique(x_combinada, return_index=True)
            poblacion_sin_repetidos = poblacion_combinada[indices_unicos]
            aptitudes_sin_repetidos = aptitudes_combinadas[indices_unicos]
            
            # 4. Eliminar sobrantes manteniendo los mejores hasta el tamaño de población
            if evolucion.minimizar:
                indices_ordenados = np.argsort(aptitudes_sin_repetidos) # Ordenar de mayor a menor
            else:
                indices_ordenados = np.argsort(aptitudes_sin_repetidos)[::-1]  # Ordenar descendente

            if len(indices_ordenados) > cantidad_poblacion:
                indices_seleccionados = indices_ordenados[:cantidad_poblacion]
            else: 

                # Tomar solo los mejores hasta alcanzar el tamaño de población deseado
                if len(indices_ordenados) > cantidad_poblacion:
                    indices_seleccionados = indices_ordenados[:cantidad_poblacion]
                else:
                    # Si no hay suficientes individuos únicos, completar con algunos aleatorios
                    indices_seleccionados = indices_ordenados
                    num_faltantes = cantidad_poblacion - len(indices_seleccionados)
                    if num_faltantes > 0:
                        indices_aleatorios = np.random.choice(len(valores_x_filtrados), 
                                                            size=num_faltantes, 
                                                            replace=False)
                        poblacion_sin_repetidos = np.append(poblacion_sin_repetidos, indices_aleatorios)
                        indices_seleccionados = np.arange(len(poblacion_sin_repetidos))
                
                # Actualizar la población con los individuos seleccionados
                evolucion.poblacion = poblacion_sin_repetidos[indices_seleccionados]

        evolucion.fig.canvas.draw()
        save_frame(evolucion.fig, frame, evolucion.temp_dir, evolucion.frame_files)

        if frame == num_generaciones -1:
            generate_video(evolucion.frame_files)
            
        evolucion.fig.tight_layout()
        return []

    # Crear canvas y animación
    canvas = FigureCanvasTkAgg(evolucion.fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Iniciar animación
    evolucion.animation = FuncAnimation(
        evolucion.fig, animate, init_func=init,
        frames=num_generaciones, interval=200, blit=True
    )


ventana, entries, frame_grafica, resultado_vars = setup_gui.setup_gui(ejecutar_algoritmo)

entry_intervalo_inicio = entries["Inicio del intervalo (A):"]
entry_intervalo_fin = entries["Fin del intervalo (B):"]
entry_resolucion = entries["Resolución (Delta x):"]
entry_cantidad_poblacion = entries["Cantidad de población:"]
entry_probabilidad_cruza = entries["Probabilidad de cruza:"]
entry_probabilidad_mutacion = entries["Probabilidad de mutación:"]
entry_probabilidad_mutacion_bits = entries["Probabilidad de mutación de bits:"]
entry_num_generaciones = entries["Cantidad de iteraciones:"]

ventana.mainloop()