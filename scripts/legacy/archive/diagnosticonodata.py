import numpy as np
import rasterio

# [especulado] Valor padrão de NoData que vimos no seu diagnóstico anterior
nodata_extremo = -1.7976931348623157e+308

print("--- Teste de Integridade de Conversão ---")

# Criamos um array pequeno com o valor problemático (float64)
teste_array = np.array([nodata_extremo, 1000.0, 2500.0], dtype='float64')

print(f"Array original (float64): {teste_array[0]}")

# [especulado] O momento em que o Python gera o Warning:
print("\nTentando converter para float32 (o que o Rasterio faz internamente)...")
try:
    with np.errstate(all='warn'): # Força o warning a aparecer se existir
        convertido = teste_array.astype('float32')
        print(f"Resultado da conversão: {convertido[0]}")
except Exception as e:
    print(f"Erro capturado: {e}")

print("\n--- Fim do Teste ---")