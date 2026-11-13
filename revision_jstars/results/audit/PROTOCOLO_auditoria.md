# Auditoria visual cega — JSTARS (R1)

300 polígonos sorteados: 10 classes × 3 faixas de pureza × 10. **Você não vê a classe atribuída nem a pureza**, de propósito: isso evita viés.

## Como fazer
1. Abra `auditoria_poligonos.kml` no Google Earth Pro (ou o `.gpkg` no QGIS sobre imagem de alta resolução). Cada polígono tem só o ID (A001…A300).
2. Ative as **imagens históricas** do Google Earth e compare a imagem mais próxima de **2023** (data do Sentinel e dos embeddings) com a mais próxima de **2021** (ano-base do MapBiomas).
3. **Não consulte** MapBiomas nem mapas de classificação durante a auditoria.
4. Preencha `auditoria_formulario_cego.csv`:
   - `classe_visual`: classe dominante que você vê no polígono. Use exatamente um destes nomes: Mata, Eucalipto, Pastagem, Agricultura, Formação Campestre, Solo Exposto, Mineração, Área Urbana, Afloramento Rochoso, Rios e Lagoas, Indeterminado.
   - `mistura_visual`: quanto a classe dominante ocupa do polígono: `>80`, `50-80` ou `<50`.
   - `mudanca_2021_2023`: `s` se o uso mudou entre as duas imagens, senão `n`.
   - `observacao`: livre (por exemplo "nuvem", "sombra", "sem imagem boa").
5. Tempo estimado: 1–2 min por polígono, de 5 a 10 horas no total. Dá para fazer em blocos (A001–A100 etc.).

## Depois
Me mande o formulário preenchido. Só então eu uso a chave (`chave_NAO_ABRIR_ate_terminar/`) para comparar a sua classe com o rótulo atribuído e com o MapBiomas, e classificar as divergências (objeto misto, erro de ponto, erro do MapBiomas, mudança temporal).
