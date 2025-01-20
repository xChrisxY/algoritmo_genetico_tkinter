import tkinter as tk
from tkinter import ttk

def create_main_window():
    """Crea y configura la ventana principal y sus widgets."""
    ventana = tk.Tk()
    ventana.title("Algoritmo Genético")
    
    # Configuración de estilos
    style = ttk.Style()
    style.configure("Result.TLabel", font=('Arial', 10), padding=5)
    style.configure("ResultValue.TLabel", font=('Arial', 10, 'bold'), padding=5)
    
    return ventana

def create_parameter_frame(ventana, ejecutar_algoritmo):
    """Crea y configura el frame de parámetros."""
    frame_parametros = ttk.Frame(ventana, padding="5")
    frame_parametros.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    parametros = [
        ("Inicio del intervalo (A):", "5"),
        ("Fin del intervalo (B):", "10"),
        ("Resolución (Delta x):", "0.1"),
        ("Cantidad de población:", "150"),
        ("Probabilidad de cruza:", "0.8"),
        ("Probabilidad de mutación:", "0.1"),
        ("Probabilidad de mutación de bits:", "0.5"),
        ("Cantidad de iteraciones:", "15")
    ]

    entries = {}
    for i, (label_text, default_value) in enumerate(parametros):
        ttk.Label(frame_parametros, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
        entry = ttk.Entry(frame_parametros)
        entry.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        entry.insert(tk.END, default_value)
        entries[label_text] = entry

    ttk.Button(frame_parametros, text="Ejecutar Algoritmo", command=ejecutar_algoritmo).grid(
        row=len(parametros), column=0, columnspan=2, pady=10
    )

    return entries

def create_right_frame(ventana):
    """Crea y configura el frame derecho con gráfica y resultados."""
    frame_derecho = ttk.Frame(ventana, padding="5")
    frame_derecho.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    frame_grafica = ttk.Frame(frame_derecho)
    frame_grafica.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    frame_resultados = ttk.LabelFrame(frame_derecho, text="Resultados del Algoritmo", padding="10")
    frame_resultados.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)

    resultados_labels = [
        ("Cantidad de puntos:", "cantidad_puntos"),
        ("Número de bits necesarios:", "bits"),
        ("Δx:", "delta_x"),
        ("Mejor valor de x:", "mejor_x"),
        ("Mejor valor de f(x):", "mejor_fx"),
        ("Representación binaria:", "mejor_binario"),
        ("Valor decimal:", "mejor_decimal")
    ]

    resultado_vars = {}
    for i, (label, var_name) in enumerate(resultados_labels):
        ttk.Label(frame_resultados, text=label, style="Result.TLabel").grid(
            row=i, column=0, sticky=tk.W, padx=5
        )
        resultado_vars[var_name] = tk.StringVar()
        ttk.Label(frame_resultados, textvariable=resultado_vars[var_name], 
                 style="ResultValue.TLabel").grid(row=i, column=1, sticky=tk.W, padx=5)

    frame_derecho.columnconfigure(0, weight=1)
    frame_resultados.columnconfigure(1, weight=1)

    return frame_grafica, resultado_vars

def setup_gui(ejecutar_algoritmo):
    """Configura toda la interfaz gráfica."""
    ventana = create_main_window()
    entries = create_parameter_frame(ventana, ejecutar_algoritmo)
    frame_grafica, resultado_vars = create_right_frame(ventana)

    ventana.columnconfigure(1, weight=1)
    ventana.rowconfigure(0, weight=1)

    return ventana, entries, frame_grafica, resultado_vars