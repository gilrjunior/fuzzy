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
        self.regras = [
            ctrl.Rule(A['ideal'] & D['ideal_ou_excesso'] & N['na_media_ou_acima'], R['baixo']),
            ctrl.Rule(A['calor_extremo'] & D['seca_severa'] & N['muito_abaixo_media'], R['critico']),
            ctrl.Rule(D['seca_severa'], R['alto']),
            ctrl.Rule(N['muito_abaixo_media'], R['alto']),
            ctrl.Rule(A['calor_moderado'] & D['deficit_leve'], R['moderado']),
        ]

    def simular(self, anomalia_temp: float, deficit_hid: float, anomalia_ndvi: float) -> float:
        self.simulador.input['anomalia_termica'] = anomalia_temp
        self.simulador.input['deficit_hidrico']  = deficit_hid
        self.simulador.input['anomalia_ndvi']    = anomalia_ndvi
        self.simulador.compute()
        return float(self.simulador.output['risco_quebra_safra'])


def main():
    ar = AgroRiskFuzzy()
    print("Exemplo → risco =", ar.simular( 2, 80, 0.1 ))

if __name__ == "__main__":
    main()
