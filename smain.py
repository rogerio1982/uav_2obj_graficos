import math
import numpy as np
import matplotlib.pyplot as plt
from calcchannel_atgg import calculate_channel
from users import Users as BaseUsers
from uavs import Uavs


import os

# --- Gerando usuários ---
np.random.seed(42)
H0 = 150
Cob0 = H0 * 1.5


users = []
for i in range(100):
    x = np.random.uniform(0, 1000)
    y = np.random.uniform(0, 1000)
    u = BaseUsers(i, x, y, 0, np.random.uniform(1e6, 5e6), 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, None)
    # atributos adicionais usados pelo cálculo/plots
    setattr(u, 'height', 1.6)
    setattr(u, 'RB_needed', 0)
    setattr(u, 'RB_allocated', 0)
    setattr(u, 'DR_served', 0.0)
    users.append(u)

# --- gerando  ---

initial_uav = Uavs(1, 379, 327, 40, 2.4, 10e6, 0, 0, True, 100, 100, False, "False", 0, Cob0, H0, 0, 0, 0)
alluav = [initial_uav]

# --- Funções ---
def compute_throughputs(uav, alluav, users):
    tps = []
    for u in users:
        DR, CQI, SINR, PRX, I, RB_needed = calculate_channel(u, uav, alluav)
        u.DR = DR
        u.CQI = CQI
        u.SINR = SINR
        u.PRX = PRX
        u.Int = I
        u.C = 1 if DR > 0 else 0
        u.RB_needed = RB_needed
        u.RB_allocated = 0
        u.DR_served = 0.0
        tps.append(DR / 1e6)
    return tps

def allocate_RBs_simple(uav, users):
    """
    Aloca RBs sequencialmente até acabar.
    Usuários que não recebem RB são marcados como não atendidos (C=0).
    """
    remaining = uav.PRB
    for u in users:
        if u.DR <= 0:
            u.C = 0
            continue
        # só aloca se houver RB suficiente
        if remaining >= u.RB_needed:
            u.RB_allocated = u.RB_needed
            u.DR_served = min(u.R_DR, u.RB_allocated * (u.DR / uav.PRB))
            u.C = 1  # atendido
            remaining -= u.RB_allocated
        else:
            u.RB_allocated = 0
            u.DR_served = 0
            u.C = 0  # não atendido
    return remaining


def mean_throughput(uav, alluav, users):
    return np.mean(compute_throughputs(uav, alluav, users))

def connected_users(users):
    return sum(u.C for u in users)

print(f"Cobertura inicial (altura {initial_uav.H} m): {initial_uav.Cob:.2f} m")

# calcula métricas
compute_throughputs(initial_uav, alluav, users)

# aloca RBs simples
remaining = allocate_RBs_simple(initial_uav, users)

mean_tp = np.mean([u.DR_served/1e6 for u in users if u.DR>0])
connected = connected_users(users)

print(f"Throughput médio inicial: {mean_tp:.2f} Mbps")
print(f"Usuários conectados: {connected} de {len(users)}\n")


for u in users:
    if u.C == 1:
        print(f"User {u.ID}: Troughput={u.DR/1e6:.2f} Mbps, CQI={u.CQI}, SINR={u.SINR:.2f} dB, "
              f"PRX={u.PRX} dBm, R_DR = {u.R_DR/1e6:.2f} Mbps, RB_needed={u.RB_needed}, RB_alloc={u.RB_allocated}, DR_served={u.DR_served/1e6:.2f} Mbps")


# --- Gráficos (adaptados do main2.py) ---
# Criar pasta 'image' se não existir
if not os.path.exists('image'):
    os.makedirs('image')

# Gráfico 4: Posição 2D dos Usuários e UAVs
plt.figure(figsize=(8, 6))
try:
    x_conn = [u.X for u in users if u.C == 1]
    y_conn = [u.Y for u in users if u.C == 1]
    x_disc = [u.X for u in users if u.C == 0]
    y_disc = [u.Y for u in users if u.C == 0]

    plt.scatter(x_disc, y_disc, c='red', label='Usuários desconectados', alpha=0.6)
    plt.scatter(x_conn, y_conn, c='blue', label='Usuários conectados', alpha=0.6)
    plt.scatter(initial_uav.X, initial_uav.Y, c='green', s=120, marker='^', label=f'UAV (h={initial_uav.H:.1f}m)')

    circle = plt.Circle((initial_uav.X, initial_uav.Y), initial_uav.Cob, color='green', fill=False, linestyle='--', alpha=0.3)
    plt.gca().add_patch(circle)

    plt.title("Posição 2D dos Usuários e UAV (Snapshot)")
    plt.xlabel("Posição X (m)")
    plt.ylabel("Posição Y (m)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('image/4-2d_positions.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Erro ao gerar gráfico 2D: {str(e)}")

# Gráfico 5: Visualização 3D dos UAVs e Usuários
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
plt.figure(figsize=(8, 6))
try:
    ax = plt.subplot(111, projection='3d')
    user_x = [u.X for u in users]
    user_y = [u.Y for u in users]
    user_z = [0] * len(users)
    ax.scatter(user_x, user_y, user_z, c='blue', label='Usuários', alpha=0.6)

    ax.scatter(initial_uav.X, initial_uav.Y, initial_uav.H, c='red', marker='^', s=100, label=f'UAV (h={initial_uav.H:.1f}m)')

    ax.set_title("Visualização 3D dos UAVs e Usuários")
    ax.set_xlabel("Posição X (m)")
    ax.set_ylabel("Posição Y (m)")
    ax.set_zlabel("Altura (m)")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, max(200, initial_uav.H * 1.2))
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig('image/5-3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Erro ao gerar gráfico 3D: {str(e)}")

# Gráfico 6: Posição 2D dos Usuários e UAVs com cores (por UAV)
plt.figure(figsize=(8, 6))
try:
    # Como há um único UAV, usamos uma única cor
    color = 'tab:orange'

    # Usuários atendidos por este UAV
    connected_users_pts = [u for u in users if u.C == 1]
    if connected_users_pts:
        user_x = [u.X for u in connected_users_pts]
        user_y = [u.Y for u in connected_users_pts]
        plt.scatter(user_x, user_y, c=color, alpha=0.6, label='Usuários conectados')

    # Usuários não conectados
    not_connected = [u for u in users if u.C == 0]
    if not_connected:
        user_x_nc = [u.X for u in not_connected]
        user_y_nc = [u.Y for u in not_connected]
        plt.scatter(user_x_nc, user_y_nc, c='gray', alpha=0.5, label='Usuários desconectados')

    # UAV
    plt.scatter(initial_uav.X, initial_uav.Y, c=[color], marker='^', s=100, label=f'UAV (h={initial_uav.H:.1f}m)')

    plt.title("Visualização 2D dos UAVs e Usuários Conectados")
    plt.xlabel("Posição X (m)")
    plt.ylabel("Posição Y (m)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('image/6-2d_colored.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"Erro ao gerar gráfico 2D adicional: {str(e)}")

'''

# --- Gráfico ---
fig, ax = plt.subplots(figsize=(8, 8))

x_conn = [u.X for u in users if u.C == 1]
y_conn = [u.Y for u in users if u.C == 1]
x_disc = [u.X for u in users if u.C == 0]
y_disc = [u.Y for u in users if u.C == 0]

ax.scatter(x_disc, y_disc, c='red', label="Usuários desconectados", alpha=0.6)
ax.scatter(x_conn, y_conn, c='blue', label="Usuários conectados", alpha=0.6)
ax.scatter(initial_uav.X, initial_uav.Y, c='green', s=150, marker='^', label="UAV")

circle = plt.Circle((initial_uav.X, initial_uav.Y), initial_uav.Cob, color='green', fill=False, linestyle='--')
ax.add_patch(circle)

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Usuários cobertos pelo UAV")
ax.legend()
plt.grid(True)
plt.show()
'''