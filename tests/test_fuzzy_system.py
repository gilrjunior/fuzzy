from projeto_fuzzy.fuzzy_system import AgroRiskFuzzy

def test_cenario_extremo():
    sistema = AgroRiskFuzzy()
    risco = sistema.simular(12, 210, -0.2)
    assert risco >= 90
