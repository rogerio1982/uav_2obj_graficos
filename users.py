class Users:
    def __init__(self, ID, X, Y, DR, R_DR, EB, PRB, CQI, SINR, PRX, C, V, M, ES, EBC, Int, cebida):
        self.ID = ID
        self.X = X
        self.Y = Y
        self.DR = DR #Taxa de dados
        self.R_DR = R_DR #taxa de dados requerida
        self.EB = EB #Estação base
        self.PRB = PRB #Total de PRBs
        self.CQI = CQI #Indicador de Qualidade do Canal
        self.SINR = SINR #Relação Sinal/Ruido
        self.PRX = PRX #potencia recebida
        self.C = C #Usuario conectado?
        self.V = V #Velocidade
        self.M = M #Momento.
        self.ES = ES #% 1 = Micro || 2 == Macro
        self.EBC = EBC #%Smalls Candidatas para conexão
        self.Int = Int #Interferência
        self.cebida = cebida
