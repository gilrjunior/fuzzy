import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class AgroRiskFuzzy:
    """
    Sistema de Inferência Fuzzy para avaliar risco de quebra de safra.
    """

    def __init__(self):
        self._definir_variaveis()
        self._definir_funcoes_pertinencia()
        self._definir_regras()
        self.sistema   = ctrl.ControlSystem(self.regras)
        self.simulador = ctrl.ControlSystemSimulation(self.sistema)

    def _definir_variaveis(self):
        self.anomalia_termica   = ctrl.Antecedent(np.arange(-15, 16, 1), 'anomalia_termica')
        self.deficit_hidrico    = ctrl.Antecedent(np.arange(0, 301, 1),    'deficit_hidrico')
        self.anomalia_ndvi      = ctrl.Antecedent(np.arange(-0.4, 0.41, 0.01), 'anomalia_ndvi')
        self.risco_quebra_safra = ctrl.Consequent(np.arange(0, 101, 1),     'risco_quebra_safra')

    def _definir_funcoes_pertinencia(self):
        at = self.anomalia_termica.universe
        self.anomalia_termica['frio_prejudicial'] = fuzz.trapmf(at, [-15, -15, -10, -5])
        self.anomalia_termica['ideal']            = fuzz.trimf(at, [-7, 0, 7])
        self.anomalia_termica['calor_moderado']   = fuzz.trimf(at, [1, 2, 3])
        self.anomalia_termica['calor_extremo']    = fuzz.trapmf(at, [11, 14, 15, 15])

        dh = self.deficit_hidrico.universe
        self.deficit_hidrico['ideal_ou_excesso']  = fuzz.trapmf(dh, [0, 0, 50, 100])
        self.deficit_hidrico['deficit_leve']      = fuzz.trimf(dh, [50, 100, 150])
        self.deficit_hidrico['deficit_moderado']  = fuzz.trimf(dh, [100, 150, 200])
        self.deficit_hidrico['seca_severa']       = fuzz.trapmf(dh, [150, 225, 300, 300])

        an = self.anomalia_ndvi.universe
        self.anomalia_ndvi['muito_abaixo_media'] = fuzz.trapmf(an, [-0.4, -0.4, -0.25, -0.15])
        self.anomalia_ndvi['abaixo_media']       = fuzz.trimf(an, [-0.2, -0.1, 0])
        self.anomalia_ndvi['na_media_ou_acima']  = fuzz.trapmf(an, [-0.05, 0.05, 0.4, 0.4])

        rq = self.risco_quebra_safra.universe
        self.risco_quebra_safra['baixo']    = fuzz.trimf(rq, [0, 15, 30])
        self.risco_quebra_safra['moderado'] = fuzz.trimf(rq, [30, 47.5, 65])
        self.risco_quebra_safra['alto']     = fuzz.trimf(rq, [60, 77.5, 95])
        self.risco_quebra_safra['critico']  = fuzz.trapmf(rq, [90, 97.5, 100, 100])

    def _definir_regras(self):
        A, D, N, R = (
            self.anomalia_termica,
            self.deficit_hidrico,
            self.anomalia_ndvi,
            self.risco_quebra_safra
        )

        # Linha 1: Déficit Hídrico = ideal_ou_excesso
        regra_1 = ctrl.Rule(N['na_media_ou_acima'] & D['ideal_ou_excesso'] & A['frio_prejudicial'], R['moderado'])
        regra_2 = ctrl.Rule(N['na_media_ou_acima'] & D['ideal_ou_excesso'] & A['ideal'],             R['baixo'])
        regra_3 = ctrl.Rule(N['na_media_ou_acima'] & D['ideal_ou_excesso'] & A['calor_moderado'],   R['baixo'])
        regra_4 = ctrl.Rule(N['na_media_ou_acima'] & D['ideal_ou_excesso'] & A['calor_extremo'],    R['moderado'])

        # Linha 2: Déficit Hídrico = deficit_leve
        regra_5 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_leve'] & A['frio_prejudicial'], R['moderado'])
        regra_6 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_leve'] & A['ideal'],             R['baixo'])
        regra_7 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_leve'] & A['calor_moderado'],   R['moderado'])
        regra_8 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_leve'] & A['calor_extremo'],    R['alto'])

        # Linha 3: Déficit Hídrico = deficit_moderado
        regra_9 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_moderado'] & A['frio_prejudicial'], R['alto'])
        regra_10 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_moderado'] & A['ideal'],             R['moderado'])
        regra_11 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_moderado'] & A['calor_moderado'],   R['alto'])
        regra_12 = ctrl.Rule(N['na_media_ou_acima'] & D['deficit_moderado'] & A['calor_extremo'],    R['critico'])
        
        # Linha 4: Déficit Hídrico = seca_severa
        regra_13 = ctrl.Rule(N['na_media_ou_acima'] & D['seca_severa'] & A['frio_prejudicial'], R['alto'])
        regra_14 = ctrl.Rule(N['na_media_ou_acima'] & D['seca_severa'] & A['ideal'],             R['alto'])
        regra_15 = ctrl.Rule(N['na_media_ou_acima'] & D['seca_severa'] & A['calor_moderado'],   R['critico'])
        regra_16 = ctrl.Rule(N['na_media_ou_acima'] & D['seca_severa'] & A['calor_extremo'],    R['critico'])

        # NDVI abaixo da média com condições favoráveis ainda representa risco moderado
        regra_17 = ctrl.Rule(N['abaixo_media'] & D['ideal_ou_excesso'] & A['ideal'], R['moderado'])
        regra_18 = ctrl.Rule(N['abaixo_media'] & D['deficit_leve'] & A['ideal'], R['alto'])
        regra_19 = ctrl.Rule(N['abaixo_media'] & D['deficit_moderado'] & A['ideal'], R['alto'])
        regra_20 = ctrl.Rule(N['abaixo_media'] & D['seca_severa'] & A['ideal'], R['critico'])
        
        # NDVI abaixo da média com calor
        regra_21 = ctrl.Rule(N['abaixo_media'] & D['ideal_ou_excesso'] & A['calor_moderado'], R['alto'])
        regra_22 = ctrl.Rule(N['abaixo_media'] & D['ideal_ou_excesso'] & A['calor_extremo'], R['alto'])
        regra_23 = ctrl.Rule(N['abaixo_media'] & D['deficit_leve'] & A['calor_moderado'], R['alto'])
        regra_24 = ctrl.Rule(N['abaixo_media'] & D['deficit_leve'] & A['calor_extremo'], R['critico'])
        
        # NDVI abaixo da média com frio
        regra_25 = ctrl.Rule(N['abaixo_media'] & D['ideal_ou_excesso'] & A['frio_prejudicial'], R['alto'])
        regra_26 = ctrl.Rule(N['abaixo_media'] & D['deficit_leve'] & A['frio_prejudicial'], R['alto'])

        # NDVI muito abaixo da média indica problemas graves independente das outras condições
        regra_27 = ctrl.Rule(N['muito_abaixo_media'] & D['ideal_ou_excesso'] & A['ideal'], R['alto'])
        regra_28 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_leve'] & A['ideal'], R['alto'])
        regra_29 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_moderado'] & A['ideal'], R['critico'])
        regra_30 = ctrl.Rule(N['muito_abaixo_media'] & D['seca_severa'] & A['ideal'], R['critico'])
        
        # NDVI muito abaixo com calor extremo
        regra_31 = ctrl.Rule(N['muito_abaixo_media'] & D['ideal_ou_excesso'] & A['calor_extremo'], R['critico'])
        regra_32 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_leve'] & A['calor_extremo'], R['critico'])
        regra_33 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_moderado'] & A['calor_extremo'], R['critico'])
        regra_34 = ctrl.Rule(N['muito_abaixo_media'] & D['seca_severa'] & A['calor_extremo'], R['critico'])
        
        # NDVI muito abaixo com calor moderado
        regra_35 = ctrl.Rule(N['muito_abaixo_media'] & D['ideal_ou_excesso'] & A['calor_moderado'], R['alto'])
        regra_36 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_leve'] & A['calor_moderado'], R['critico'])
        regra_37 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_moderado'] & A['calor_moderado'], R['critico'])
        regra_38 = ctrl.Rule(N['muito_abaixo_media'] & D['seca_severa'] & A['calor_moderado'], R['critico'])
        
        # NDVI muito abaixo com frio
        regra_39 = ctrl.Rule(N['muito_abaixo_media'] & D['ideal_ou_excesso'] & A['frio_prejudicial'], R['alto'])
        regra_40 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_leve'] & A['frio_prejudicial'], R['alto'])
        regra_41 = ctrl.Rule(N['muito_abaixo_media'] & D['deficit_moderado'] & A['frio_prejudicial'], R['critico'])
        regra_42 = ctrl.Rule(N['muito_abaixo_media'] & D['seca_severa'] & A['frio_prejudicial'], R['critico'])

        # Regra de fallback para cenários extremos
        regra_43 = ctrl.Rule(A['calor_extremo'] & D['seca_severa'], R['critico'])
        regra_44 = ctrl.Rule(A['frio_prejudicial'] & D['seca_severa'], R['alto'])
        regra_45 = ctrl.Rule(D['seca_severa'], R['alto'])
        regra_46 = ctrl.Rule(N['muito_abaixo_media'], R['alto'])
        regra_47 = ctrl.Rule(A['ideal'] & D['ideal_ou_excesso'] & N['na_media_ou_acima'], R['baixo'])

        # Compilar todas as regras
        self.regras = [
            regra_1, regra_2, regra_3, regra_4,
            regra_5, regra_6, regra_7, regra_8,
            regra_9, regra_10, regra_11, regra_12,
            regra_13, regra_14, regra_15, regra_16,
            regra_17, regra_18, regra_19, regra_20,
            regra_21, regra_22, regra_23, regra_24,
            regra_25, regra_26,
            regra_27, regra_28, regra_29, regra_30,
            regra_31, regra_32, regra_33, regra_34,
            regra_35, regra_36, regra_37, regra_38,
            regra_39, regra_40, regra_41, regra_42,
            regra_43, regra_44, regra_45, regra_46, regra_47
        ]

    def simular(self, anomalia_temp: float, deficit_hid: float, anomalia_ndvi: float) -> float:
        self.simulador.input['anomalia_termica'] = anomalia_temp
        self.simulador.input['deficit_hidrico']  = deficit_hid
        self.simulador.input['anomalia_ndvi']    = anomalia_ndvi
        self.simulador.compute()
        return float(self.simulador.output['risco_quebra_safra'])

    def categorizar(self, risco: float) -> str:

        if risco < 30:
            return "baixo"
        elif risco < 60:
            return "moderado"
        elif risco < 90:
            return "alto"
        else:
            return "crítico"


def main():

    ar = AgroRiskFuzzy()
    print("Risco de quebra de safra =", ar.categorizar(ar.simular(0, 50, 0.1 )))

    #    (0, 50, 0.1),      # Ideal
    #    (2, 100, -0.05),   # Moderado
    #    (5, 250, -0.3),    # Alto
    #    (12, 280, -0.35),  # Crítico
    #    (-12, 0, -0.4),    # Frio extremo + NDVI crítico
    #    (15, 300, 0.4),    # Calor extremo + seca severa


if __name__ == "__main__":
    main()
