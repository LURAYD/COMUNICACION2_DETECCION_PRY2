# Diagnóstico terminado

Escenarios completados: id, cfo_zero, cfo_shift, multipath_shift, both_shift, beta_025, beta_035, beta_050, beta_065

Finalizados9 escenarios,720 padres,6480 versiones y25920 predicciones. Auditoría aprobada en verification.json; modelos intactos en completed.json. Reporte: ../../docs/DIAGNOSTICO_CANAL_RESULTADOS.md. No quedan cálculos pendientes.

Reanudar/reconstruir tablas: `.\.venv\Scripts\python.exe -m experiments.exp09_channel_diagnostic`. Se reutilizan predicciones atómicas por escenario; no se entrena. No implementar política, ADC ni GNU Radio como continuación automática. No hay apagado programado.
