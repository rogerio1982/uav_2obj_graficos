from users import Users
from uavs import Uavs
#from calc import calculate_channel
from calcchannel_atgg import calculate_channel
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import os
import csv

# Parâmetro de separação mínima entre UAVs (em metros)
MIN_UAV_SEPARATION = 200.0  # Aumentar separação mínima para 200m

# Configuração do problema
uav = 6 # número de UAVs
s = 100   # número de usuários (será atualizado dinamicamente ao ler o CSV)
resBlo = 100  # resource blocks
areax = 1000
areay = 1000
t = 40   # tempo de simulação
hmin = 30  # altura mínima dos UAVs
hmax = 100    # altura máxima dos UAVs

# Configuração do NSGA-II
POPULATION_SIZE = 100  # Dobrar população para mais diversidade
P_CROSSOVER = 0.7      # Crossover para exploração
P_MUTATION = 0.3       # Mutação para diversidade (soma = 1.0)
MAX_GENERATIONS = 100  # Dobrar gerações para mais convergência

# Variável global para armazenar os usuários lidos do CSV
global_users = None

# Função auxiliar para converter números complexos para float
def safe_float(value):
    if isinstance(value, complex):
        return float(value.real)
    return float(value)

def convert_latlon_to_xy(lat, lon, ref_lat, ref_lon, scale_factor=1000):
    """
    Converte coordenadas lat/lon para coordenadas X,Y em metros
    ref_lat, ref_lon: ponto de referência (centro da área)
    scale_factor: fator de escala para converter graus em metros
    """
    # Conversão aproximada: 1 grau ≈ 111320 metros
    x = (lon - ref_lon) * 111320 * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * 111320
    
    return x, y

def read_users_from_csv(filename='points_inside.csv'):
    """
    Lê posições dos usuários do arquivo CSV com coordenadas lat/lon
    """
    global global_users
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
        try:
            setattr(us, 'height', 1.6)
        except Exception:
            pass
        setattr(us, 'original_lat', lat)
        setattr(us, 'original_lon', lon)
        users.append(us)
    
    global_users = users
    return users

# Criar tipos para o DEAP
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # Maximizar ambos os objetivos
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Inicializar toolbox
toolbox = base.Toolbox()

def create_individual():
    """Cria um indivíduo (posições dos UAVs) baseado na distribuição dos usuários"""
    individual = creator.Individual()
    
    if global_users is not None:
        # Estratégia inteligente: posicionar UAVs em clusters de usuários
        # Encontrar clusters de usuários usando k-means simples
        user_positions = [(user.X, user.Y) for user in global_users]
        
        # Dividir usuários em clusters baseados na densidade
        clusters = []
        used_users = set()
        
        for i, (x, y) in enumerate(user_positions):
            if i in used_users:
                continue
                
            # Criar cluster ao redor deste usuário
            cluster = [i]
            used_users.add(i)
            
            # Encontrar usuários próximos (dentro de 150m)
            for j, (x2, y2) in enumerate(user_positions):
                if j != i and j not in used_users:
                    dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist <= 150:  # Raio de cluster
                        cluster.append(j)
                        used_users.add(j)
            
            clusters.append(cluster)
        
        # Posicionar UAVs nos clusters com verificação de separação mínima
        uav_count = 0
        placed_positions = []  # Armazenar posições já colocadas
        
        for cluster in clusters:
            if uav_count >= uav:
                break
                
            if len(cluster) >= 3:  # Clusters com pelo menos 3 usuários
                # Calcular centro do cluster
                cluster_x = sum(user_positions[i][0] for i in cluster) / len(cluster)
                cluster_y = sum(user_positions[i][1] for i in cluster) / len(cluster)
                
                # Tentar posicionar UAV próximo ao centro do cluster
                for attempt in range(10):  # Máximo 10 tentativas por cluster
                    x = cluster_x + random.uniform(-80, 80)  # Aumentar raio de busca
                    y = cluster_y + random.uniform(-80, 80)
                    h = random.uniform(hmin, hmax)
                    
                    # Garantir que está dentro da área
                    x = max(0, min(areax, x))
                    y = max(0, min(areay, y))
                    
                    # Verificar separação mínima dos UAVs já colocados
                    valid_position = True
                    for px, py in placed_positions:
                        dist = math.sqrt((x - px)**2 + (y - py)**2)
                        if dist < MIN_UAV_SEPARATION:
                            valid_position = False
                            break
                    
                    if valid_position:
                        # Garantir que a altura esteja dentro dos limites
                        if h > hmax:
                            h = hmax
                        elif h < hmin:
                            h = hmin
                            
                        individual.extend([x, y, h])
                        placed_positions.append((x, y))
                        uav_count += 1
                        break
        
        # Se não temos UAVs suficientes, adicionar posições aleatórias próximas aos usuários
        while uav_count < uav:
            reference_user = random.choice(global_users)
            
            # Tentar múltiplas posições até encontrar uma válida
            for attempt in range(15):  # Máximo 15 tentativas
                radius = random.uniform(50, 150)  # Aumentar raio mínimo
                angle = random.uniform(0, 2 * math.pi)
                
                x = reference_user.X + radius * math.cos(angle)
                y = reference_user.Y + radius * math.sin(angle)
                h = random.uniform(hmin, hmax)
                
                x = max(0, min(areax, x))
                y = max(0, min(areay, y))
                
                # Verificar separação mínima dos UAVs já colocados
                valid_position = True
                for px, py in placed_positions:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist < MIN_UAV_SEPARATION:
                        valid_position = False
                        break
                
                if valid_position:
                    if h > hmax:
                        h = hmax
                    elif h < hmin:
                        h = hmin
                        
                    individual.extend([x, y, h])
                    placed_positions.append((x, y))
                    uav_count += 1
                    break
    else:
        # Fallback: posições aleatórias se não há usuários globais
        for _ in range(uav):
            x = random.uniform(0, areax)
            y = random.uniform(0, areay)
            h = random.uniform(hmin, hmax)
            if h > hmax:
                h = hmax
            elif h < hmin:
                h = hmin
            individual.extend([x, y, h])
    
    return individual

def evaluate(individual):
    """Avalia um indivíduo retornando os dois objetivos"""
    try:
        # Garantir separação mínima entre UAVs (ajuste local de x,y)
        coords = individual[:]
        # ajusta somente x,y, mantendo h
        for i in range(uav):
            xi = coords[i*3]
            yi = coords[i*3+1]
            # clamp em limites
            coords[i*3] = max(0.0, min(areax, safe_float(xi)))
            coords[i*3+1] = max(0.0, min(areay, safe_float(yi)))
        # empurra UAVs muito próximos
        changed = True
        iter_guard = 0
        while changed and iter_guard < 50:
            changed = False
            iter_guard += 1
            for i in range(uav):
                xi = coords[i*3]
                yi = coords[i*3+1]
                for j in range(i+1, uav):
                    xj = coords[j*3]
                    yj = coords[j*3+1]
                    dx = safe_float(xj) - safe_float(xi)
                    dy = safe_float(yj) - safe_float(yi)
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6:
                        # separa aleatoriamente se coincidentes
                        angle = random.random() * 2*math.pi
                        coords[j*3] = max(0.0, min(areax, safe_float(xj) + math.cos(angle)*MIN_UAV_SEPARATION))
                        coords[j*3+1] = max(0.0, min(areay, safe_float(yj) + math.sin(angle)*MIN_UAV_SEPARATION))
                        changed = True
                    elif dist < MIN_UAV_SEPARATION:
                        # move j para afastar até min sep mantendo nos limites
                        push = (MIN_UAV_SEPARATION - dist) / dist
                        nx = safe_float(xj) + dx * push
                        ny = safe_float(yj) + dy * push
                        nx = max(0.0, min(areax, nx))
                        ny = max(0.0, min(areay, ny))
                        if abs(nx - safe_float(xj)) > 1e-6 or abs(ny - safe_float(yj)) > 1e-6:
                            coords[j*3] = nx
                            coords[j*3+1] = ny
                            changed = True

        # Usar usuários globais lidos do CSV
        if global_users is None:
            # Fallback: criar usuários aleatórios se não foram carregados do CSV
            alluser = []
            for i in range(s):
                x_u = random.uniform(0, areax)
                y_u = random.uniform(0, areay)
                demand_bps = random.uniform(1e6, 5e6)
                us = Users(i, x_u, y_u, 0, demand_bps, 0, 1, 0, 0, 0, False, 0, 0, 1, 0, 0, 0)
                try:
                    setattr(us, 'height', 1.6)
                except Exception:
                    pass
                alluser.append(us)
        else:
            alluser = global_users

        # Criar UAVs com as posições do indivíduo
        alluav = []
        for i in range(uav):
            x = safe_float(coords[i*3])
            y = safe_float(coords[i*3 + 1])
            # Garantir que a altura esteja dentro dos limites
            h = safe_float(individual[i*3 + 2])
            if h > hmax:
                h = hmax
            elif h < hmin:
                h = hmin
            # B=10e6 Hz, PRB e PRB_F iguais a resBlo para permitir alocação efetiva
            uavs = Uavs(i, x, y, 23, 2.4, 10e6, 0, 0, True, resBlo, resBlo, False, "False", 0, h*2, h, 0, 0, 0)
            # compatibilidade com calcchannel_atgg (usa atributo 'height')
            try:
                setattr(uavs, 'height', uavs.H)
            except Exception:
                pass
            alluav.append(uavs)

        # Alocar usuários e calcular métricas (versão simplificada como greedy)
        on = 0
        total_throughput = 0
        
        # Resetar estado dos usuários
        for user in alluser:
            user.C = False
            user.EB = 0
            user.ES = 0
        
        # Resetar estado dos UAVs
        for uav_obj in alluav:
            uav_obj.PRB_F = resBlo
            uav_obj.U = 0
            uav_obj.MAX_U = 0
        
        for i in alluser:
            best_uav = None
            best_chanel = None
            best_throughput = 0
            
            # Encontrar o melhor UAV para cada usuário
            for x in alluav:
                chanel = calculate_channel(i, x, alluav)
                # calcchannel_atgg retorna (DR, CQI, SINR, PRX, I, RB_needed)
                if chanel and chanel[0] > 0:
                    throughput = safe_float(chanel[0])
                    rb_needed = int(chanel[5])
                    if throughput > best_throughput and x.PRB_F >= rb_needed:
                        best_throughput = throughput
                        best_uav = x
                        best_chanel = chanel
            
            # Conectar ao melhor UAV encontrado
            if best_uav is not None and best_chanel is not None:
                # Verificar se o usuário já está conectado
                if not i.C:  # Só conectar se não estiver já conectado
                    i.EB = best_uav.ID
                    i.ES = 1
                    i.CQI = safe_float(best_chanel[1])
                    i.SINR = safe_float(best_chanel[2])
                    # PRX já é retornado por calcchannel_atgg
                    i.PRX = safe_float(best_chanel[3])
                    i.C = True
                    # consumir PRBs necessários (não throughput)
                    best_uav.PRB_F = best_uav.PRB_F - int(best_chanel[5])
                    best_uav.U = best_uav.U + 1
                    on = on + 1
                    best_uav.MAX_U += 1
                    i.DR = safe_float(best_chanel[0])
                    total_throughput += i.DR

        # Objetivo 1: Maximizar número de usuários conectados (com peso maior)
        objective1 = float(on) * 3.0  # Aumentar peso para 3x para priorizar usuários conectados
        
        # Objetivo 2: Maximizar throughput total (em Mbps) com normalização
        objective2 = float(total_throughput / 10**6) * 0.5  # Reduzir peso do throughput

        return objective1, objective2
    except Exception as e:
        print(f"Erro na avaliação: {str(e)}")
        return 0.0, 0.0  # Retorna valores padrão em caso de erro

def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """Crossover SBX com limites"""
    size = len(ind1)
    for i in range(0, size, 3):  # Para cada UAV (x, y, h)
        if random.random() < 0.5:
            # sanitize for complex values that may appear due to numeric issues
            try:
                ind1[i] = float(ind1[i].real) if isinstance(ind1[i], complex) else float(ind1[i])
                ind2[i] = float(ind2[i].real) if isinstance(ind2[i], complex) else float(ind2[i])
            except Exception:
                pass
            if abs(ind1[i] - ind2[i]) > 1e-14:
                if ind1[i] > ind2[i]:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
                beta = 1.0 + (2.0 * (ind2[i] - low) / (ind1[i] - ind2[i]))
                alpha = 2.0 - beta**(-(eta + 1.0))
                rand = random.random()
                if rand <= 1.0/alpha:
                    betaq = (rand * alpha)**(1.0/(eta + 1.0))
                else:
                    betaq = (1.0/(2.0 - rand * alpha))**(1.0/(eta + 1.0))
                ind1[i] = 0.5 * ((1 + betaq) * ind2[i] + (1 - betaq) * ind1[i])
                ind2[i] = 0.5 * ((1 - betaq) * ind2[i] + (1 + betaq) * ind1[i])
                
                # Aplicar o mesmo para y e h
                for j in range(1, 3):
                    # sanitize values
                    try:
                        ind1[i+j] = float(ind1[i+j].real) if isinstance(ind1[i+j], complex) else float(ind1[i+j])
                        ind2[i+j] = float(ind2[i+j].real) if isinstance(ind2[i+j], complex) else float(ind2[i+j])
                    except Exception:
                        pass
                    if abs(ind1[i+j] - ind2[i+j]) > 1e-14:
                        if ind1[i+j] > ind2[i+j]:
                            ind1[i+j], ind2[i+j] = ind2[i+j], ind1[i+j]
                        # Ajustar limites para altura
                        if j == 2:  # Se for altura
                            low_bound = hmin
                            up_bound = hmax
                        else:  # Se for x ou y
                            low_bound = low
                            up_bound = up
                            
                        beta = 1.0 + (2.0 * (ind2[i+j] - low_bound) / (ind1[i+j] - ind2[i+j]))
                        alpha = 2.0 - beta**(-(eta + 1.0))
                        rand = random.random()
                        if rand <= 1.0/alpha:
                            betaq = (rand * alpha)**(1.0/(eta + 1.0))
                        else:
                            betaq = (1.0/(2.0 - rand * alpha))**(1.0/(eta + 1.0))
                        ind1[i+j] = 0.5 * ((1 + betaq) * ind2[i+j] + (1 - betaq) * ind1[i+j])
                        ind2[i+j] = 0.5 * ((1 - betaq) * ind2[i+j] + (1 + betaq) * ind1[i+j])
                        
                        # Garantir que a altura fique dentro dos limites e seja um número real
                        if j == 2:
                            ind1[i+j] = float(max(hmin, min(hmax, abs(ind1[i+j]))))
                            ind2[i+j] = float(max(hmin, min(hmax, abs(ind2[i+j]))))

    return ind1, ind2

def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Mutação polinomial com limites que mantém UAVs próximos aos usuários"""
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            y = individual[i]
            
            # Se for posição X ou Y, usar mutação baseada em usuários
            if (i % 3) == 0 or (i % 3) == 1:  # X ou Y
                if global_users is not None:
                    # Mutação baseada em usuários próximos
                    uav_index = i // 3
                    current_x = individual[uav_index * 3]
                    current_y = individual[uav_index * 3 + 1]
                    
                    # Encontrar usuário mais próximo
                    min_dist = float('inf')
                    nearest_user = None
                    for user in global_users:
                        dist = math.sqrt((safe_float(current_x) - safe_float(user.X))**2 + (safe_float(current_y) - safe_float(user.Y))**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_user = user
                    
                    if nearest_user is not None and random.random() < 0.97:  # 97% chance de mover em direção ao usuário
                        # Mover em direção ao usuário com verificação de separação mínima
                        for attempt in range(5):  # Máximo 5 tentativas
                            if (i % 3) == 0:  # X
                                direction = 1 if safe_float(nearest_user.X) > safe_float(current_x) else -1
                                new_x = safe_float(current_x) + direction * random.uniform(30, 120)
                                new_y = safe_float(current_y)
                            else:  # Y
                                direction = 1 if safe_float(nearest_user.Y) > safe_float(current_y) else -1
                                new_x = safe_float(current_x)
                                new_y = safe_float(current_y) + direction * random.uniform(30, 120)
                            
                            # Verificar separação mínima dos outros UAVs
                            valid_position = True
                            for j in range(0, len(individual), 3):
                                if j != uav_index * 3:  # Não verificar contra o próprio UAV
                                    other_x = safe_float(individual[j])
                                    other_y = safe_float(individual[j + 1])
                                    dist = math.sqrt((new_x - other_x)**2 + (new_y - other_y)**2)
                                    if dist < MIN_UAV_SEPARATION:
                                        valid_position = False
                                        break
                            
                            if valid_position:
                                y = new_x if (i % 3) == 0 else new_y
                                break
                    else:
                        # Mutação normal
                        low_bound = low
                        up_bound = up
                        delta1 = (y - low_bound) / (up_bound - low_bound)
                        delta2 = (up_bound - y) / (up_bound - low_bound)
                        rand = random.random()
                        mut_pow = 1.0 / (eta + 1.)
                        if rand < 0.5:
                            xy = 1.0 - delta1
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(eta + 1.0))
                            deltaq = val**mut_pow - 1.0
                        else:
                            xy = 1.0 - delta2
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(eta + 1.0))
                            deltaq = 1.0 - val**mut_pow
                        y = y + deltaq * (up_bound - low_bound)
                else:
                    # Fallback: mutação normal
                    low_bound = low
                    up_bound = up
                    delta1 = (y - low_bound) / (up_bound - low_bound)
                    delta2 = (up_bound - y) / (up_bound - low_bound)
                    rand = random.random()
                    mut_pow = 1.0 / (eta + 1.)
                    if rand < 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(eta + 1.0))
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(eta + 1.0))
                        deltaq = 1.0 - val**mut_pow
                    y = y + deltaq * (up_bound - low_bound)
            else:  # Altura
                low_bound = hmin
                up_bound = hmax
                delta1 = (y - low_bound) / (up_bound - low_bound)
                delta2 = (up_bound - y) / (up_bound - low_bound)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy**(eta + 1.0))
                    deltaq = val**mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy**(eta + 1.0))
                    deltaq = 1.0 - val**mut_pow
                y = y + deltaq * (up_bound - low_bound)
            
            # sanitize possible complex due to numerical operations
            try:
                y = float(y.real) if isinstance(y, complex) else float(y)
            except Exception:
                pass
            individual[i] = y
            
            # Garantir que a altura fique dentro dos limites
            if (i % 3) == 2:
                individual[i] = float(max(hmin, min(hmax, abs(individual[i]))))
    return individual,

def local_search(individual, toolbox):
    """Aplica busca local para melhorar um indivíduo"""
    if not global_users:
        return False
    
    original_fitness = individual.fitness.values
    improved = False
    
    # Tentar pequenos ajustes em cada UAV
    for uav_idx in range(uav):
        x_idx = uav_idx * 3
        y_idx = uav_idx * 3 + 1
        h_idx = uav_idx * 3 + 2
        
        # Encontrar usuários próximos a este UAV
        uav_x = safe_float(individual[x_idx])
        uav_y = safe_float(individual[y_idx])
        
        nearby_users = []
        for user in global_users:
            dist = math.sqrt((safe_float(user.X) - uav_x)**2 + (safe_float(user.Y) - uav_y)**2)
            if dist < 300:  # Raio de busca local
                nearby_users.append((user, dist))
        
        if not nearby_users:
            continue
            
        # Ordenar usuários por proximidade
        nearby_users.sort(key=lambda x: x[1])
        
        # Tentar mover UAV em direção aos usuários mais próximos
        for user, dist in nearby_users[:3]:  # Testar com os 3 usuários mais próximos
            # Calcular direção para o usuário
            dx = safe_float(user.X) - uav_x
            dy = safe_float(user.Y) - uav_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Normalizar direção
                dx /= distance
                dy /= distance
                
                # Tentar diferentes distâncias
                for step in [20, 40, 60]:
                    new_x = uav_x + dx * step
                    new_y = uav_y + dy * step
                    new_h = individual[h_idx] + random.uniform(-10, 10)
                    
                    # Verificar limites
                    new_x = max(0, min(areax, new_x))
                    new_y = max(0, min(areay, new_y))
                    new_h = max(hmin, min(hmax, new_h))
                    
                    # Criar indivíduo temporário
                    temp_individual = individual[:]
                    temp_individual[x_idx] = new_x
                    temp_individual[y_idx] = new_y
                    temp_individual[h_idx] = new_h
                    
                    # Avaliar
                    temp_fitness = toolbox.evaluate(temp_individual)
                    
                    # Verificar se é melhor (considerando ambos os objetivos)
                    if (temp_fitness[0] > original_fitness[0] or 
                        (temp_fitness[0] == original_fitness[0] and temp_fitness[1] > original_fitness[1])):
                        # Aplicar melhoria
                        individual[x_idx] = new_x
                        individual[y_idx] = new_y
                        individual[h_idx] = new_h
                        individual.fitness.values = temp_fitness
                        original_fitness = temp_fitness
                        improved = True
                        break
                
                if improved:
                    break
    
    return improved

def main():
    global s
    # Carregar usuários do CSV
    print("Lendo posições dos usuários do arquivo points_inside.csv...")
    users = read_users_from_csv('points_inside.csv')
    s = len(users)  # Atualiza o número de usuários com o valor real do CSV
    print(f"Carregados {len(users)} usuários do arquivo CSV")
    
    # Configuração do problema
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cxSimulatedBinaryBounded, low=0, up=1000, eta=20)
    toolbox.register("mutate", mutPolynomialBounded, low=0, up=1000, eta=20, indpb=1.0/9)
    toolbox.register("select", tools.selNSGA2)
    
    # Criar população inicial
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Inicializar indivíduos com valores aleatórios dentro dos limites
    for ind in pop:
        for i in range(0, len(ind), 3):
            ind[i] = random.uniform(0, areax)  # Posição X
            ind[i+1] = random.uniform(0, areay)  # Posição Y
            # Garantir que a altura inicial esteja dentro dos limites
            h = random.uniform(hmin, hmax)
            if h > hmax:
                h = hmax
            elif h < hmin:
                h = hmin
            ind[i+2] = h  # Altura (entre hmin e hmax)
    
    # Estatísticas para acompanhar a evolução
    stats = tools.Statistics(lambda ind: ind.fitness.values if ind.fitness.valid else (0.0, 0.0))
    stats.register("avg_throughput", lambda x: sum(v[1] for v in x) / len(x) if x else 0.0)
    stats.register("avg_users", lambda x: sum(v[0] / 2.0 for v in x) / len(x) if x else 0.0)
    
    # Logbook para registrar estatísticas
    logbook = tools.Logbook()
    
    # Avaliar população inicial
    print("Avaliando população inicial...")
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Registrar estatísticas iniciais
    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    print(f"Geração 0: Throughput médio = {record['avg_throughput']:.2f} Mbps, Usuários médios = {record['avg_users']:.2f}")
    
    # Evolução
    for gen in range(1, MAX_GENERATIONS + 1):
        print(f"\nGeração {gen}:")
        
        # Criar e avaliar offspring
        offspring = algorithms.varOr(pop, toolbox, lambda_=POPULATION_SIZE, cxpb=P_CROSSOVER, mutpb=P_MUTATION)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        if invalid_ind:
            print(f"Avaliando {len(invalid_ind)} indivíduos...")
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
        # Selecionar próxima geração
        pop = toolbox.select(pop + offspring, POPULATION_SIZE)
        
        # Busca local nos melhores indivíduos (a cada 10 gerações)
        if gen % 10 == 0:
            print("Aplicando busca local...")
            # Selecionar os 10 melhores indivíduos para busca local
            best_individuals = tools.selBest(pop, 10)
            
            for ind in best_individuals:
                # Aplicar busca local simples
                improved = local_search(ind, toolbox)
                if improved:
                    # Reavaliar se houve melhoria
                    ind.fitness.values = toolbox.evaluate(ind)
        
        # Registrar estatísticas
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        
        print(f"Throughput médio = {record['avg_throughput']:.2f} Mbps")
        print(f"Usuários médios = {record['avg_users']:.2f}")
    
    # Plotar resultados
    print("\nGerando gráficos...")

    # Criar pasta 'image_nsga' se não existir
    out_dir = 'image_nsga'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Gráfico 1: Evolução do Throughput
    plt.figure(figsize=(8, 6))
    gen_data = logbook.select("gen")
    throughput_data = logbook.select("avg_throughput")
    plt.plot(gen_data, throughput_data)
    plt.title("Evolução do Throughput Médio")
    plt.xlabel("Geração")
    plt.ylabel("Throughput Médio (Mbps)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1-throughput_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 2: Evolução de Usuários Conectados
    plt.figure(figsize=(8, 6))
    users_data = logbook.select("avg_users")
    plt.plot(gen_data, users_data)
    plt.title("Evolução de Usuários Conectados")
    plt.xlabel("Geração")
    plt.ylabel("Número Médio de Usuários")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '2-users_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 3: Frente de Pareto
    plt.figure(figsize=(8, 6))
    try:
        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        front_fitnesses = [ind.fitness.values for ind in front if ind.fitness.valid]
        if front_fitnesses:
            front_throughput = [f[1] for f in front_fitnesses]
            front_users = [f[0] for f in front_fitnesses]
            plt.scatter(front_throughput, front_users, c='red', label='Frente de Pareto')
    except Exception as e:
        print(f"Erro ao gerar frente de Pareto: {str(e)}")

    plt.title("Frente de Pareto")
    plt.xlabel("Throughput (Mbps)")
    plt.ylabel("Usuários Conectados")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3-pareto_front.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Gráfico 4: Posição 2D dos Usuários e UAVs
    plt.figure(figsize=(8, 6))
    try:
        # Encontrar indivíduo com melhor número de usuários conectados (objetivo 1)
        best_users_ind = max(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else 0.0)
        best_throughput_ind = best_users_ind
        
        # Criar usuários e UAVs da melhor solução
        # Usar usuários globais do CSV para plotagem
        if global_users is not None:
            alluser = global_users
        else:
            # Fallback: criar usuários aleatórios se não foram carregados do CSV
            alluser = []
            for i in range(s):
                taxa = 35
                us = Users(i, random.randint(1, areax), random.randint(1, areay), taxa, 1, 0, t, 0, 0, 0, False, 0, 0, 1, 0, 0, 0)
                try:
                    setattr(us, 'height', 1.6)
                except Exception:
                    pass
                alluser.append(us)
        
        # aplicar a mesma separação mínima para plotagem
        coords = list(best_throughput_ind)
        for i in range(uav):
            coords[i*3] = max(0.0, min(areax, float(coords[i*3])))
            coords[i*3+1] = max(0.0, min(areay, float(coords[i*3+1])))
        changed = True
        iter_guard = 0
        while changed and iter_guard < 50:
            changed = False
            iter_guard += 1
            for i in range(uav):
                xi = coords[i*3]
                yi = coords[i*3+1]
                for j in range(i+1, uav):
                    xj = coords[j*3]
                    yj = coords[j*3+1]
                    dx = safe_float(xj) - safe_float(xi)
                    dy = safe_float(yj) - safe_float(yi)
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6:
                        angle = random.random() * 2*math.pi
                        coords[j*3] = max(0.0, min(areax, xj + math.cos(angle)*MIN_UAV_SEPARATION))
                        coords[j*3+1] = max(0.0, min(areay, yj + math.sin(angle)*MIN_UAV_SEPARATION))
                        changed = True
                    elif dist < MIN_UAV_SEPARATION:
                        push = (MIN_UAV_SEPARATION - dist) / dist
                        nx = xj + dx * push
                        ny = yj + dy * push
                        nx = max(0.0, min(areax, nx))
                        ny = max(0.0, min(areay, ny))
                        if abs(nx - xj) > 1e-6 or abs(ny - yj) > 1e-6:
                            coords[j*3] = nx
                            coords[j*3+1] = ny
                            changed = True

        # Plotar usuários
        user_x = [us.X for us in alluser]
        user_y = [us.Y for us in alluser]
        plt.scatter(user_x, user_y, c='blue', label='Usuários', alpha=0.6)
        
        # Plotar UAVs e suas áreas de cobertura
        for i in range(uav):
            x = coords[i*3]
            y = coords[i*3+1]
            h = best_throughput_ind[i*3+2]
            
            # Calcular raio de cobertura baseado na altura (aproximação)
            coverage_radius = h * 2  # Aproximação: raio = 2x altura
            
            # Plotar área de cobertura (círculo)
            circle = plt.Circle((x, y), coverage_radius, color='red', fill=False, linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)
            
            # Plotar UAV
            plt.scatter(x, y, c='red', marker='^', s=100, label=f'UAV {i+1} (h={h:.1f}m)')
        
        # Calcular métricas reais da melhor solução
        print(f"\n=== MÉTRICAS DA MELHOR SOLUÇÃO (Máximo de Usuários) ===")
        print(f"Fitness: {best_users_ind.fitness.values}")
        print(f"Usuários conectados: {best_users_ind.fitness.values[0] / 2.0:.0f}")  # Dividir por 2 pois foi multiplicado por 2 na função evaluate
        print(f"Throughput: {best_users_ind.fitness.values[1]:.2f} Mbps")
        
        plt.title(f"Posição 2D dos Usuários e UAVs\n(Melhor Solução: {best_users_ind.fitness.values[0] / 2.0:.0f} usuários conectados)")
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
        
    except Exception as e:
        print(f"Erro ao gerar gráfico 2D: {str(e)}")
        import traceback
        traceback.print_exc()

    # Gráfico 5: Visualização 3D dos UAVs e Usuários
    plt.figure(figsize=(8, 6))
    try:
        ax = plt.subplot(111, projection='3d')
        # Plotar usuários no solo (z=0)
        user_x = [us.X for us in alluser]
        user_y = [us.Y for us in alluser]
        user_z = [0] * len(alluser)  # Usuários no solo
        ax.scatter(user_x, user_y, user_z, c='blue', label='Usuários', alpha=0.6)
        
        # Plotar UAVs com suas alturas
        for i in range(uav):
            x = coords[i*3]
            y = coords[i*3+1]
            h = best_users_ind[i*3+2]
            ax.scatter(x, y, h, c='red', marker='^', s=100, label=f'UAV {i+1} (h={h:.1f}m)')
        
        # Configurar o gráfico 3D
        ax.set_title("Visualização 3D dos UAVs e Usuários")
        ax.set_xlabel("Posição X (m)")
        ax.set_ylabel("Posição Y (m)")
        ax.set_zlabel("Altura (m)")
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        
        # Ajustar limites dos eixos
        ax.set_xlim(0, areax)
        ax.set_ylim(0, areay)
        ax.set_zlim(0, max([best_users_ind[i*3+2] for i in range(uav)]) * 1.2)
        
        # Ajustar ângulo de visualização
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5-3d_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Erro ao gerar gráfico 3D: {str(e)}")

    
    
    # Snapshot adicional: replicar cenário do smain.py e imprimir métricas de usuários conectados
    try:
        np.random.seed(42)
        demo_users = []
        for i in range(100):
            x = np.random.uniform(0, 1000)
            y = np.random.uniform(0, 1000)
            us = Users(i, x, y, 0, np.random.uniform(1e6, 5e6), 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            setattr(us, 'height', 1.6)
            setattr(us, 'RB_needed', 0)
            setattr(us, 'RB_allocated', 0)
            setattr(us, 'DR_served', 0.0)
            demo_users.append(us)

        H0 = 150
        Cob0 = H0 * 1.5
        demo_uav = Uavs(1, 379, 327, 40, 2.4, 10e6, 0, 0, True, 100, 100, False, "False", 0, Cob0, H0, 0, 0, 0)
        setattr(demo_uav, 'height', demo_uav.H)
        demo_alluav = [demo_uav]

        # calcular canais
        for u in demo_users:
            DR, CQI, SINR, PRX, I, RB_needed = calculate_channel(u, demo_uav, demo_alluav)
            u.DR = DR
            u.CQI = CQI
            u.SINR = SINR
            u.PRX = PRX
            u.Int = I
            u.C = 1 if DR > 0 else 0
            u.RB_needed = RB_needed
            u.RB_allocated = 0
            u.DR_served = 0.0

        # alocação simples de RBs
        remaining = demo_uav.PRB_F
        for u in demo_users:
            if u.DR <= 0:
                u.C = 0
                continue
            if remaining >= u.RB_needed:
                u.RB_allocated = u.RB_needed
                # proporcional à fração de RBs alocados, limitado à demanda
                u.DR_served = min(u.R_DR, u.RB_allocated * (u.DR / demo_uav.PRB))
                u.C = 1
                remaining -= u.RB_allocated
            else:
                u.RB_allocated = 0
                u.DR_served = 0.0
                u.C = 0

        # imprimir métricas de usuários conectados (formato smain)
        for u in demo_users:
            if u.C == 1:
                print(f"User {u.ID}: Troughput={u.DR/1e6:.2f} Mbps, CQI={u.CQI}, SINR={u.SINR:.2f} dB, PRX={u.PRX} dBm, R_DR = {u.R_DR/1e6:.2f} Mbps, RB_needed={u.RB_needed}, RB_alloc={u.RB_allocated}, DR_served={u.DR_served/1e6:.2f} Mbps")
    except Exception as e:
        print(f"Erro no snapshot do smain: {str(e)}")

if __name__ == "__main__":
    main()
