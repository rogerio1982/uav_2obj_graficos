import math

def calculate_channel(U, S, alluav):
    # Distância 3D entre UAV e usuário
    D_3d = math.sqrt((U.X - S.X) ** 2 + (U.Y - S.Y) ** 2 + (S.H - U.height) ** 2)
    d_2d = math.sqrt((U.X - S.X) ** 2 + (U.Y - S.Y) ** 2)  # Distância horizontal

    if D_3d <= S.Cob and S.D:

        # Ruído branco
        #white_noise = 7.4e-13
        white_noise = 7.4e-12  # maior que o valor anterior

        # Interferência de outras células
        I = 0

        # Frequência (GHz)
        f = 2.4

        # Alturas
        h_UT = 1.6      # Altura do usuário (m)
        h_BS = S.H      # Altura da estação base (m)
        h_UAV = U.height # Altura do UAV

        # Modelo ATG 3GPP TR 36.777 air-to-ground
        # Probabilidade de LoS (Line-of-Sight)
        p_LOS = 1 / (1 + 0.16 * math.exp(-0.03 * (math.degrees(math.atan((h_UAV - h_UT) / d_2d)) - 0.03)))

        # Path loss LoS e NLoS
        PL_LOS = 28 + 22 * math.log10(d_2d) + 20 * math.log10(f)
        PL_NLOS = 13.54 + 39.08 * math.log10(d_2d) + 20 * math.log10(f) - 0.6 * (h_UAV - h_UT)

        # Path loss efetiva
        PL = p_LOS * PL_LOS + (1 - p_LOS) * PL_NLOS

        # Ganho de antena UAV e usuário (simples)
        G_BS = 2.4
        G_UT = 0

        # Potência recebida (W)
        Prx = 10 ** ((S.RP + G_BS + G_UT - PL) / 10) / 1000

        # Interferência de outros UAVs
        for small in alluav:
            if ((small.D and small.ID) != S.ID):
                d_interf = math.sqrt((small.X - U.X)**2 + (small.Y - U.Y)**2 + (small.height - h_UT)**2)
                PL_interf = 28 + 22 * math.log10(d_interf) + 20 * math.log10(f)
                I += 10 ** ((small.RP - PL_interf) / 10) / 1000

        SINRw = Prx / (white_noise + I)
        SINR = 10 * math.log10(SINRw)

        # largura de banda por RB
        C = S.B / S.PRB

        # taxa por RB
        DR_per_RB = C * math.log2(1 + SINRw)

        # taxa total alcançável com todos os RBs
        DR = DR_per_RB * S.PRB

        # RBs necessários para atender a taxa requerida do usuário
        RB_needed = math.ceil(U.R_DR / DR_per_RB)

        # CQI aproximado (em função do SINR em dB)
        CQI = max(0, min(15, round(0.5 * (SINR + 5))))

        # Potência recebida em dBm
        PRX = round(10 * math.log10(1000 * Prx))

    else:
        SINR = 0
        DR = 0
        CQI = 0
        I = 0
        PRX = 0
        RB_needed = 0

    return DR, CQI, SINR, PRX, I, RB_needed
