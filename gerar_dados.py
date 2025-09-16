from shapely.geometry import Polygon, Point
import random
import csv

# Definição das coordenadas do polígono (x = longitude, y = latitude)
coords = [
    (-49.1439509, -5.3534992),
    (-49.1394448, -5.3545246),
    (-49.1320633, -5.3523028),
    (-49.1312051, -5.3497819),
    (-49.1257118, -5.3458936),
    (-49.1233944, -5.3425607),
    (-49.123609, -5.3383305),
    (-49.1306471, -5.3418343),
    (-49.1352391, -5.3467054),
    (-49.1406464, -5.3510637),
    (-49.1439509, -5.3534992)
]

# Cria o objeto polígono
poly = Polygon(coords)

# Bounding box
minx, miny, maxx, maxy = poly.bounds

# Fixar a semente para reprodutibilidade
random.seed(42)

points_inside = []

# Gera até ter 100 pontos dentro
while len(points_inside) < 100:
    random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
    if poly.contains(random_point):
        points_inside.append(random_point)

# Salva os pontos em CSV
with open('points_inside.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['lon', 'lat'])  # Cabeçalho
    for pt in points_inside:
        writer.writerow([pt.x, pt.y])

print("Arquivo 'points_inside.csv' gerado com sucesso.")
