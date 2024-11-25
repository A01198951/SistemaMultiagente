
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Función para leer los puntos desde un archivo
def read_points_from_txt(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # Separar las coordenadas por coma y tomar los dos primeros valores (X, Y)
            values = line.strip().split(',')
            if len(values) >= 2:
                x, y = map(float, values[:2])
                points.append((x, y))
    return points

# Dimensiones del robot
ROBOT_WIDTH = 0.16  # metros
ROBOT_HEIGHT = 0.20  # metros
ROBOT_RADIUS = max(ROBOT_WIDTH, ROBOT_HEIGHT) / 2

# Leer puntos desde archivos TXT
points1 = read_points_from_txt('XY_302_6_1_R.txt')
points2 = read_points_from_txt('XY_302_6_2_R.txt')

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Obstáculos originales
obstacles = [
    {
        'x': [-0.218391135768374, -0.746320221915368, -0.441520221915368, 0.0864088642316258],
        'y': [0.225864543073497, -0.0789354569265029, -0.606864543073497, -0.302064543073497]
    },
    {
        'x': [0.441520221915368, -0.0864088642316257, 0.218391135768374, 0.746320221915368],
        'y': [0.606864543073497, 0.302064543073497, -0.225864543073497, 0.0789354569265031]
    },
    {
        'x': [-0.6952, -1.3048, -1.3048, -0.6952],
        'y': [1.3048, 1.3048, 0.6952, 0.6952]
    },
    {
        'x': [1.3048, 0.6952, 0.6952, 1.3048],
        'y': [-0.6952, -0.6952, -1.3048, -1.3048]
    },
    {
        'x': [0.873302236800156, 0.28447385309434, 0.126697763199844, 0.71552614690566],
        'y': [1.43552614690566, 1.59330223680016, 1.00447385309434, 0.846697763199844]
    },
    {
        'x': [-0.568947706188681, -1, -1.43105229381132, -1],
        'y': [-1, -0.568947706188681, -1, -1.43105229381132]
    }
]

# Graficar obstáculos originales e inflados
for obstacle in obstacles:
    coords = list(zip(obstacle['x'], obstacle['y']))
    poly = Polygon(coords)
    inflated_poly = poly.buffer(ROBOT_RADIUS)

    x_original, y_original = poly.exterior.xy
    ax.fill(x_original, y_original, color='gray', alpha=0.5, label='Obstáculo Original' if obstacles.index(obstacle) == 0 else "")

    x_inflated, y_inflated = inflated_poly.exterior.xy
    ax.fill(x_inflated, y_inflated, color='lightgray', alpha=0.3, label='Obstáculo Inflado' if obstacles.index(obstacle) == 0 else "")

# Graficar rutas desde los puntos leídos
points1_x, points1_y = zip(*points1)
points2_x, points2_y = zip(*points2)

ax.plot(points1_x, points1_y, 'r-', label='Camino 1')
ax.plot(points2_x, points2_y, 'b-', label='Camino 2')

# Posiciones objetivo
targets_x = [
    -0.86717069892473, -0.277318548387096, 0.286122311827957,
    -1.01683467741935, 0.673487903225808, -1.37778897849462, 1.54506048387097
]
targets_y = [
    -0.356552419354838, 0.550235215053764, -0.497412634408602,
    1.52745295698925, 0.629469086021506, -1.36898521505376, -0.999227150537633
]
ax.scatter(targets_x, targets_y, c='green', marker='x', s=100, label='Posiciones Objetivo')

# Configurar leyendas y etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Rutas, Obstáculos y Posiciones Objetivo')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# Mostrar el gráfico
plt.legend()
plt.show()

'''
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Posiciones iniciales
initial_positions_x = [-1.5, -2]
initial_positions_y = [-2, 0]

# Obstáculos originales
obstacles = [
    {
        'x': [-0.218391135768374, -0.746320221915368, -0.441520221915368, 0.0864088642316258],
        'y': [0.225864543073497, -0.0789354569265029, -0.606864543073497, -0.302064543073497]
    },
    {
        'x': [0.441520221915368, -0.0864088642316257, 0.218391135768374, 0.746320221915368],
        'y': [0.606864543073497, 0.302064543073497, -0.225864543073497, 0.0789354569265031]
    },
    {
        'x': [-0.6952, -1.3048, -1.3048, -0.6952],
        'y': [1.3048, 1.3048, 0.6952, 0.6952]
    },
    {
        'x': [1.3048, 0.6952, 0.6952, 1.3048],
        'y': [-0.6952, -0.6952, -1.3048, -1.3048]
    },
    {
        'x': [0.873302236800156, 0.28447385309434, 0.126697763199844, 0.71552614690566],
        'y': [1.43552614690566, 1.59330223680016, 1.00447385309434, 0.846697763199844]
    },
    {
        'x': [-0.568947706188681, -1, -1.43105229381132, -1],
        'y': [-1, -0.568947706188681, -1, -1.43105229381132]
    }
]

# Posiciones objetivo
targets_x = [
    -0.86717069892473, -0.277318548387096, 0.286122311827957,
    -1.01683467741935, 0.673487903225808, -1.37778897849462, 1.54506048387097
]
targets_y = [
    -0.356552419354838, 0.550235215053764, -0.497412634408602,
    1.52745295698925, 0.629469086021506, -1.36898521505376, -0.999227150537633
]

# Puntos para ordered_points1
points1 = [
    (-1.4998457988624532, -1.6232498274505343),
    (-1.37778897849462, -1.36898521505376),
    (-1.4998457988624532, -1.6232498274505343),
    (-1.5, -2),
    (-0.9998505218333515, -1.981455399053473),
    (-0.507317860879311, -1.9739927829784116),
    (0.00014003222485126088, -1.9739927829784116),
    (0.5075979253290148, -1.981455399053473),
    (1.007593202358116, -1.981455399053473),
    (1.5001258633121566, -1.9739927829784116),
    (1.507588479387218, -1.4963853541744936),
    (1.54506048387097, -0.999227150537633),
    (1.5374389436874623, -0.6829602019928207),
    (1.5001258633121566, -0.47400695189110653),
    (1.007593202358116, -0.47400695189110653),
    (0.6046119343048106, -0.4963948001162901),
    (0.286122311827957, -0.497412634408602),
    (0.6046119343048106, -0.4963948001162901),
    (1.007593202358116, -0.47400695189110653),
    (1.5001258633121566, -0.47400695189110653),
    (1.5001258633121566, -0.13818922851335147),
    (1.4852006311620336, 0.2722546556150154),
    (1.5001258633121566, 0.6080723789927704),
    (1.0747567470336667, 0.630460227217954),
    (0.673487903225808, 0.629469086021506),
    (1.0747567470336667, 0.630460227217954),
    (1.5001258633121566, 0.6080723789927704),
    (1.495710672022863, 1.036962027129571),
    (1.5037657400611355, 1.504155973349358),
    (1.4876556039845914, 1.8344137629185169),
    (0.9343534306234993, 1.8806190759831294),
    (0.48284857552738014, 1.885062532940363),
    (0.00915046272579989, 1.8895110778254094),
    (-0.36383593162186045, 1.8939647164641409),
    (-0.3385135719527783, 1.575695735549215),
    (-0.33002073723856995, 1.1935181734098776),
    (-0.313035067810155, 0.8707904542699922),
    (-0.277318548387096, 0.550235215053764),
    (-0.313035067810155, 0.8707904542699922),
    (-0.33002073723856995, 1.1935181734098776),
    (-0.3385135719527783, 1.575695735549215),
    (-0.36383593162186045, 1.8939647164641409),
    (-0.7801409770915679, 1.9154091241175148),
    (-1.0052143294472913, 1.9540508390283797),
    (-1.01683467741935, 1.52745295698925),
    (-1.0052143294472913, 1.9540508390283797),
    (-1.487203621338388, 1.9844500303208008),
    (-1.9661586126259856, 1.989012377683764),
    (-1.976849613773921, 1.6107739630277087),
    (-2.0116501579630928, 1.1496667525211883),
    (-1.991374828802761, 0.7320602222511323),
    (-2.0000749648500538, 0.3318539640756617),
    (-1.991374828802761, -0.3293563755185933),
    (-1.643369386911048, -0.3119561034240079),
    (-1.2953639450193348, -0.3032559673767148),
    (-0.86717069892473, -0.356552419354838)
]

# Puntos para ordered_points2
points2 = [
    (-1.6779861350563783, 0.005913318230731157),
    (-1.267177665104497, 0.005913318230731157),
    (-1.2430124609896804, -0.30823433526188415),
    (-0.86717069892473, -0.356552419354838),
    (-1.2430124609896804, -0.30823433526188415),
    (-1.7263165432860115, -0.2921241991853396),
    (-1.7827020195539163, -0.6868225330606768),
    (-1.7746469515156447, -0.9365296422471148),
    (-1.7665918834773722, -1.3553931802372683),
    (-1.37778897849462, -1.36898521505376),
    (-1.4998457988624532, -1.6232498274505343),
    (-1.5, -2),
    (-0.9998505218333515, -1.981455399053473),
    (-0.507317860879311, -1.9739927829784116),
    (0.00014003222485126088, -1.9739927829784116),
    (0.5075979253290148, -1.981455399053473),
    (1.007593202358116, -1.981455399053473),
    (1.5001258633121566, -1.9739927829784116),
    (1.507588479387218, -1.4963853541744936),
    (1.54506048387097, -0.999227150537633),
    (1.5374389436874623, -0.6829602019928207),
    (1.5001258633121566, -0.47400695189110653),
    (1.007593202358116, -0.47400695189110653),
    (0.6046119343048106, -0.4963948001162901),
    (0.286122311827957, -0.497412634408602),
    (0.6046119343048106, -0.4963948001162901),
    (1.007593202358116, -0.47400695189110653),
    (1.5001258633121566, -0.47400695189110653),
    (1.5001258633121566, -0.13818922851335147),
    (1.4852006311620336, 0.2722546556150154),
    (1.5001258633121566, 0.6080723789927704),
    (1.0747567470336667, 0.630460227217954),
    (0.673487903225808, 0.629469086021506),
    (1.0747567470336667, 0.630460227217954),
    (1.5001258633121566, 0.6080723789927704),
    (1.495710672022863, 1.036962027129571),
    (1.5037657400611355, 1.504155973349358),
    (1.4876556039845914, 1.8344137629185169),
    (0.9343534306234993, 1.8806190759831294),
    (0.48284857552738014, 1.885062532940363),
    (0.00915046272579989, 1.8895110778254094),
    (-0.36383593162186045, 1.8939647164641409),
    (-0.3385135719527783, 1.575695735549215),
    (-0.33002073723856995, 1.1935181734098776),
    (-0.313035067810155, 0.8707904542699922),
    (-0.277318548387096, 0.550235215053764),
    (-0.313035067810155, 0.8707904542699922),
    (-0.33002073723856995, 1.1935181734098776),
    (-0.3385135719527783, 1.575695735549215),
    (-0.36383593162186045, 1.8939647164641409),
    (-0.7801409770915679, 1.9154091241175148),
    (-1.0052143294472913, 1.9540508390283797),
    (-1.01683467741935, 1.52745295698925),
]

# Dimensiones del robot
ROBOT_WIDTH = 0.16  # metros
ROBOT_HEIGHT = 0.20  # metros
ROBOT_RADIUS = max(ROBOT_WIDTH, ROBOT_HEIGHT) / 2

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Graficar posiciones iniciales
ax.plot(initial_positions_x, initial_positions_y, 'ro', label='Posiciones Iniciales')

# Lista para almacenar obstáculos inflados
inflated_obstacles = []

# Graficar obstáculos originales e inflados
for obstacle in obstacles:
    # Crear polígono del obstáculo original
    coords = list(zip(obstacle['x'], obstacle['y']))
    poly = Polygon(coords)
    
    # Inflar el obstáculo considerando el tamaño del robot
    inflated_poly = poly.buffer(ROBOT_RADIUS)
    inflated_obstacles.append(inflated_poly)
    
    # Extraer coordenadas para graficar el obstáculo original
    x_original, y_original = poly.exterior.xy
    ax.fill(x_original, y_original, color='gray', alpha=0.5, label='Obstáculo Original' if obstacles.index(obstacle) == 0 else "")
    
    # Extraer coordenadas para graficar el obstáculo inflado
    x_inflated, y_inflated = inflated_poly.exterior.xy
    ax.fill(x_inflated, y_inflated, color='lightgray', alpha=0.3, label='Obstáculo Inflado' if obstacles.index(obstacle) == 0 else "")

# Graficar posiciones objetivo
ax.plot(targets_x, targets_y, 'bo', label='Posiciones Objetivo')

# Configurar leyendas y etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Posiciones y Obstáculos con Inflado por Tamaño del Robot')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')


# Lista para almacenar los puntos seleccionados
clicked_points = []

# Función para manejar clics
def onclick(event):
    if event.inaxes == ax:  # Verifica si el clic está dentro de los ejes
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        print(f'{x:}, {y:}')
        # Marcar el punto en el gráfico
        ax.plot(x, y, 'gx', label='Punto Seleccionado' if len(clicked_points) == 1 else "")
        fig.canvas.draw()

# Conectar el evento de clic
fig.canvas.mpl_connect('button_press_event', onclick)

points1_x, points1_y = zip(*points1)
points2_x, points2_y = zip(*points2)

ax.plot(points1_x, points1_y, 'r-', label='Camino 1')
ax.plot(points2_x, points2_y, 'b-', label='Camino 2')

fig.canvas.draw()

# Mostrar el gráfico
plt.show()
'''