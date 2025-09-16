class Uavs:
    def __init__(self, ID, X, Y, RP, Fr, B, U, VU, D, PRB, PRB_F, F, C, I, Cob, H, Int, UB, MAX_U):
        self.ID = ID
        self.X = X #lat
        self.Y = Y #log
        self.RP = RP # Potência de transmissão 23dBm
        self.Fr = Fr #Frequência 2.4 GHz
        self.B = B #banda 20 MHz
        self.U = U #total de Usuários
        self.VU = VU #usu conectados
        self.D = D #Estação base conectada?
        self.PRB = PRB #Total de PRBs
        self.PRB_F = PRB_F #Total de PRBs disponiveis
        self.F = F
        self.C = C
        self.I = I
        self.Cob = Cob #cobe3rtura
        self.H = H #Atura
        self.Int = Int #interferencia
        self.UB = UB #Usuários bloqueados pela micro
        self.MAX_U = MAX_U
