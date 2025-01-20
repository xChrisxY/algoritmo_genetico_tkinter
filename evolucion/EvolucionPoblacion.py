class EvolucionPoblacion:
    def __init__(self, minimizar=False):
        self.generacion_actual = 0
        self.poblacion = None
        self.mejores = []
        self.peores = []
        self.promedios = []
        self.animation = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.scatter = None
        self.canvas = None
        # Variables para almacenar el mejor resultado
        self.mejor_x_final = None
        self.mejor_fx_final = None
        self.mejor_binario_final = None
        self.mejor_decimal_final = None
        # Para almacenar los frames
        self.temp_dir = None 
        self.frame_files = []
        self.animation_completed = False
        self.minimizar = minimizar