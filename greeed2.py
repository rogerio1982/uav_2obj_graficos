from users import Users
from uavs import Uavs
from calcchannel_atgg import calculate_channel
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import csv

# Configuração do problema
uav = 6  # número de UAVs a posicionar
s = 100  # número de usuários
resBlo = 100  # resource blocks
areax = 1000
areay = 1000
t = 40   # tempo de simulação (não utilizado diretamente aqui)
hmin = 30
hmax = 100

# Pesos para escalarização (prioridade de objetivos)
# w_users para usuários conectados (normalizado por s)
# w_tp para throughput total (normalizado por 1e9 = Gbps)
w_users = 0.9
w_tp = 0.1

def convert_latlon_to_xy(lat, lon, ref_lat, ref_lon, scale_factor=1000):
    """
    Converte coordenadas lat/lon para coordenadas X,Y em metros
    ref_lat, ref_lon: ponto de referência (centro da área)
    scale_factor: fator de escala para converter graus em metros
    """
    # Conversão aproximada: 1 grau ≈ 111320 metros
    x = (lon - ref_lon) * 111320 * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * 111320
    
    # Normaliza para a área desejada (0 a areax/areay)
    # Encontra os limites das coordenadas originais
    return x, y

def read_users_from_csv(filename='points_inside.csv'):
    """
    Lê posições dos usuários do arquivo CSV com coordenadas lat/lon
    """
    users = []
    
    # Calcula o centro da área baseado nas coordenadas do CSV
    lats = []
    lons = []
    
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lats.append(float(row['lat']))
            lons.append(float(row['lon']))
    
    # Calcula o centro da área
    ref_lat = (min(lats) + max(lats)) / 2
    ref_lon = (min(lons) + max(lons)) / 2
    
    # Calcula o fator de escala para normalizar para a área desejada
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    scale_x = areax / (lon_range * 111320 * math.cos(math.radians(ref_lat)))
    scale_y = areay / (lat_range * 111320)
    scale_factor = min(scale_x, scale_y)
    
    print(f"Centro da área: lat={ref_lat:.6f}, lon={ref_lon:.6f}")
    print(f"Fator de escala: {scale_factor:.2f}")
    
    # Cria os usuários com as coordenadas convertidas
    np.random.seed(42)  # Garantir reprodutibilidade
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        x, y = convert_latlon_to_xy(lat, lon, ref_lat, ref_lon, scale_factor)
        
        # Normaliza para a área desejada (0 a areax/areay)
        x = (x * scale_factor) + (areax / 2)
        y = (y * scale_factor) + (areay / 2)
        
        # Garante que as coordenadas estejam dentro da área
        x = max(0, min(areax, x))
        y = max(0, min(areay, y))
        
        us = Users(i, x, y, 0, np.random.uniform(1e6, 5e6), 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0) 
        setattr(us, 'height', 1.6)
        setattr(us, 'RB_needed', 0)
        setattr(us, 'RB_allocated', 0)
        setattr(us, 'DR_served', 0.0)
        setattr(us, 'original_lat', lat)
        setattr(us, 'original_lon', lon)
        users.append(us)
    
    return users



def clone_uavs(uavs_list: list) -> list:
    cloned = []
    for u in uavs_list:
        c = Uavs(u.ID, u.X, u.Y, u.RP, u.Fr, u.B, u.U, u.VU, u.D, u.PRB, u.PRB_F, u.F, u.C, u.I, u.Cob, u.H, u.Int, u.UB, u.MAX_U)
        setattr(c, 'height', c.H)
        cloned.append(c)
    return cloned

def build_uav(ID: int, x: float, y: float, h: float) -> Uavs:
    h = max(hmin, min(hmax, float(h)))
    cob = h * 2  # raio de cobertura aproximado
    u = Uavs(ID, x, y, 23, 2.4, 10e6, 0, 0, True, resBlo, resBlo, False, "False", 0, cob, h, 0, 0, 0)
    setattr(u, 'height', u.H)
    return u

def evaluate_set(alluav: list, users: list) -> tuple:
    # Avalia conectando usuários ao melhor UAV disponível (greedy por usuário)
    sim_uavs = clone_uavs(alluav)
    on = 0
    total_throughput = 0.0
    for u in users:
        best = None
        best_ch = None
        best_tp = 0.0
        for s_uav in sim_uavs:
            ch = calculate_channel(u, s_uav, sim_uavs)
            if ch and ch[0] > 0:
                tp = float(ch[0])
                rb_needed = int(ch[5])
                if s_uav.PRB_F >= rb_needed and tp > best_tp:
                    best_tp = tp
                    best = s_uav
                    best_ch = ch
        if best is not None and best_ch is not None:
            best.PRB_F -= int(best_ch[5])
            on += 1
            total_throughput += float(best_ch[0])
    return on, total_throughput

def greedy_place_optimized(users: list, k: int, max_iterations: int = 50):
    """Algoritmo greedy iterativo que reposiciona UAVs para maximizar objetivos"""
    # Inicializar com posições aleatórias
    placed = []
    for i in range(k):
        x = np.random.uniform(0, areax)
        y = np.random.uniform(0, areay)
        h = np.random.uniform(hmin, hmax)
        uav = build_uav(i, x, y, h)
        placed.append(uav)
    
    gen_list, thr_list, users_list = [], [], []
    
    # Calcular métricas iniciais
    initial_users, initial_throughput = evaluate_set(placed, users)
    gen_list.append(0)
    users_list.append(initial_users)
    thr_list.append(initial_throughput / 1e6)
    print(f"Iteração 0: {initial_users} usuários, {initial_throughput/1e6:.2f} Mbps")
    
    # Iterações de otimização
    for iteration in range(1, max_iterations + 1):
        improved = False
        best_improvement = 0
        best_uav_idx = -1
        best_new_uav = None
        
        # Tentar reposicionar cada UAV
        for uav_idx in range(k):
            current_users, current_throughput = evaluate_set(placed, users)
            
            # Gerar candidatos para reposicionamento
            candidates = generate_repositioning_candidates(users, placed, uav_idx)
            
            for x, y, h in candidates:
                # Criar novo UAV na posição candidata
                new_uav = build_uav(uav_idx, x, y, h)
                temp_placed = placed.copy()
                temp_placed[uav_idx] = new_uav
                
                # Avaliar nova configuração
                new_users, new_throughput = evaluate_set(temp_placed, users)
                
                # Calcular melhoria usando escalarização
                users_improvement = (new_users - current_users) / s  # Normalizar por total de usuários
                tp_improvement = (new_throughput - current_throughput) / 1e9  # Normalizar para Gbps
                improvement = w_users * users_improvement + w_tp * tp_improvement
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_uav_idx = uav_idx
                    best_new_uav = new_uav
                    improved = True
        
        # Aplicar melhor melhoria se encontrada
        if improved and best_improvement > 0:
            placed[best_uav_idx] = best_new_uav
            final_users, final_throughput = evaluate_set(placed, users)
            gen_list.append(iteration)
            users_list.append(final_users)
            thr_list.append(final_throughput / 1e6)
            print(f"Iteração {iteration}: {final_users} usuários, {final_throughput/1e6:.2f} Mbps (melhoria: {best_improvement:.2f})")
        else:
            # Sem melhoria, manter valores anteriores
            gen_list.append(iteration)
            users_list.append(users_list[-1])
            thr_list.append(thr_list[-1])
            print(f"Iteração {iteration}: Sem melhoria - {users_list[-1]} usuários, {thr_list[-1]:.2f} Mbps")
    
    return placed, gen_list, thr_list, users_list

def generate_repositioning_candidates(users: list, placed: list, uav_idx: int):
    """Gera candidatos para reposicionamento de um UAV específico"""
    candidates = []
    
    # Gerar candidatos baseados na distribuição dos usuários
    for user in users:
        for _ in range(5):  # 5 candidatos por usuário
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(50, 200)
            
            x = user.X + distance * np.cos(angle)
            y = user.Y + distance * np.sin(angle)
            h = np.random.uniform(hmin, hmax)
            
            # Garantir que está dentro da área
            x = max(0, min(areax, x))
            y = max(0, min(areay, y))
            
            # Verificar separação mínima dos outros UAVs
            valid = True
            for i, existing in enumerate(placed):
                if i != uav_idx:  # Não verificar contra o próprio UAV
                    dist = np.sqrt((x - existing.X)**2 + (y - existing.Y)**2)
                    if dist < 250.0:  # Separação mínima
                        valid = False
                        break
            
            if valid:
                candidates.append((x, y, h))
    
    return candidates[:20]  # Limitar a 20 candidatos

def evaluate_set_fast(uavs: list, users: list, already_connected: set = None):
    """Versão otimizada da avaliação - só avalia usuários não conectados"""
    if already_connected is None:
        already_connected = set()
    
    on = 0
    total_throughput = 0.0
    
    for u in users:
        if u.ID in already_connected:
            continue
            
        best_dr = 0.0
        best_uav = None
        
        for uav in uavs:
            if not uav.D:  # D indica se está ativo/conectado
                continue
                
            # Cálculo de distância 3D
            d_3d = np.sqrt((u.X - uav.X)**2 + (u.Y - uav.Y)**2 + (u.height - uav.height)**2)
            
            if d_3d <= uav.Cob:
                # Cálculo do canal
                DR, CQI, SINR, PRX, I, RB_needed = calculate_channel(u, uav, uavs)
                
                if DR > best_dr and RB_needed <= uav.PRB_F:
                    best_dr = DR
                    best_uav = uav
        
        if best_dr > 0:
            on += 1
            total_throughput += best_dr
            already_connected.add(u.ID)
    
    return on, total_throughput

def greedy_place(users: list, k: int):
    """Wrapper para manter compatibilidade - usa versão otimizada"""
    return greedy_place_optimized(users, k)

def plot_results(alluav: list, users: list, gen_list=None, thr_list=None, users_list=None):
    out_dir = 'image_greedy'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Gráfico 1: Evolução do Throughput
    if gen_list is not None and thr_list is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(gen_list, thr_list)
        plt.title("Evolução do Throughput Médio (Greedy)")
        plt.xlabel("Passo (UAVs posicionados)")
        plt.ylabel("Throughput (Mbps)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '1-throughput_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Gráfico 2: Evolução de Usuários Conectados
    if gen_list is not None and users_list is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(gen_list, users_list)
        plt.title("Evolução de Usuários Conectados (Greedy)")
        plt.xlabel("Passo (UAVs posicionados)")
        plt.ylabel("Número de Usuários")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2-users_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2D snapshot
    plt.figure(figsize=(8, 6))
    user_x = [u.X for u in users]
    user_y = [u.Y for u in users]
    plt.scatter(user_x, user_y, c='blue', label='Usuários', alpha=0.6)
    for i, s_uav in enumerate(alluav):
        circle = plt.Circle((s_uav.X, s_uav.Y), s_uav.Cob, color='red', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)
        plt.scatter(s_uav.X, s_uav.Y, c='red', marker='^', s=100, label=f'UAV {i+1} (h={s_uav.H:.1f}m)')
    plt.title("Greedy - Posição 2D dos Usuários e UAVs")
    plt.xlabel("Posição X (m)")
    plt.ylabel("Posição Y (m)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, areax)
    plt.ylim(0, areay)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4-2d_positions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3D visualization
    try:
        ax = plt.subplot(111, projection='3d')
        user_x = [u.X for u in users]
        user_y = [u.Y for u in users]
        user_z = [0] * len(users)
        ax.scatter(user_x, user_y, user_z, c='blue', label='Usuários', alpha=0.6)
        for i, s_uav in enumerate(alluav):
            ax.scatter(s_uav.X, s_uav.Y, s_uav.H, c='red', marker='^', s=100, label=f'UAV {i+1} (h={s_uav.H:.1f}m)')
        ax.set_title("Greedy - Visualização 3D dos UAVs e Usuários")
        ax.set_xlabel("Posição X (m)")
        ax.set_ylabel("Posição Y (m)")
        ax.set_zlabel("Altura (m)")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        ax.set_xlim(0, areax)
        ax.set_ylim(0, areay)
        max_h = max([s.H for s in alluav]) if alluav else hmax
        ax.set_zlim(0, max_h * 1.2)
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5-3d_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráfico 3D (Greedy): {str(e)}")
        
def main():
    print("Executando abordagem Greedy para posicionamento de UAVs...")
    print("Lendo posições dos usuários do arquivo points_inside.csv...")
    users = read_users_from_csv('points_inside.csv')
    global s
    s = len(users)  # Atualiza o número de usuários com o valor real do CSV
    print(f"Carregados {len(users)} usuários do arquivo CSV")
    alluav, gen_list, thr_list, users_list = greedy_place(users, uav)

    on, tp = evaluate_set(alluav, users)
    print(f"Usuários conectados: {on} de {len(users)}")
    print(f"Throughput total: {tp/1e6:.2f} Mbps")

    # Imprimir métricas detalhadas dos usuários conectados
    sim_uavs = clone_uavs(alluav)
    for u in users:
        best = None
        best_ch = None
        best_tp = 0.0
        for s_uav in sim_uavs:
            ch = calculate_channel(u, s_uav, sim_uavs)
            if ch and ch[0] > 0:
                tp_u = float(ch[0])
                rb_needed = int(ch[5])
                if s_uav.PRB_F >= rb_needed and tp_u > best_tp:
                    best_tp = tp_u
                    best = s_uav
                    best_ch = ch
        if best and best_ch:
            rb_needed = int(best_ch[5])
            # DR_served proporcional aos RBs alocados e limitado pela demanda do usuário
            dr_served = min(u.R_DR, rb_needed * (best_ch[0] / best.PRB))
            print(
                f"User {u.ID}: Troughput={best_ch[0]/1e6:.2f} Mbps, "
                f"CQI={best_ch[1]}, SINR={best_ch[2]:.2f} dB, PRX={best_ch[3]} dBm, "
                f"R_DR = {u.R_DR/1e6:.2f} Mbps, RB_needed={rb_needed}, RB_alloc={rb_needed}, "
                f"DR_served={dr_served/1e6:.2f} Mbps"
            )
            best.PRB_F -= rb_needed

    plot_results(alluav, users, gen_list, thr_list, users_list)

if __name__ == "__main__":
    main()