import numpy as np
import matplotlib.pyplot as plt

# Definindo os pontos Wi-Fi e suas potências
points = [(2, 10, -90), (12, 4, -60), (13, 10, -70)]
# Definindo os users
users = [(1, 13), (10, 3), (11,18), (12,11)]

# Criando uma grade para o heatmap
x = np.linspace(0, 20, 100)
y = np.linspace(0, 20, 100)
X, Y = np.meshgrid(x, y)

# Calculando a potência para cada ponto Wi-Fi na grade
heatmap = np.zeros_like(X)
for point in points:
    x0, y0, power = point
    distance = np.sqrt((X - x0)**2 + (Y - y0)**2)
    power_density = power * np.log10(distance)
    heatmap += power_density
    print (heatmap)

# Criando o gráfico
plt.figure(figsize=(10, 10))
plt.pcolormesh(X, Y, heatmap, cmap='viridis')
plt.colorbar(label='Potência (dBm)')
plt.scatter([point[0] for point in points], [point[1] for point in points], color='red', marker='x', label='UAV-BSs')
plt.scatter([user[0] for user in users], [user[1] for user in users], color='yellow', marker='x', label='Users')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap de Potência UAV-BSs')
plt.legend()
plt.show()

