# Diagnóstico exploratorio de canal y complementariedad

Diseño fijado en `configs/channel_diagnostic.json` y hashes de modelos en `results/channel_diagnostic/protocol_frozen.json`, antes de inferir. No hay entrenamiento, búsqueda, política ni GNU Radio.

La revisión de `dataset_e2.json` y `_generate_parent` confirma que el shift histórico cambia **CFO y banco multipath**, no beta. ID usa CFO uniforme [−100,100] Hz y FIR con retardos máximos 0/2/5 muestras; shift usa magnitud CFO 200–300 Hz y retardos máximos 9/13 muestras. Ambos sortean beta entre 0,25/0,35/0,50. Fs=8000 muestras/s, Rs=1000 símbolos/s. Los retardos de 13 muestras equivalen a1,625ms en este modelo discreto; no se atribuyen a una geometría RF real.

Se generan 720 padres nuevos con semilla2026110611: seis clases, SNR−5/0/10/20dB,30 por celda. Nueve versiones por padre: ID, CFO0, solo CFOshift, solo multipathshift, ambos, beta0,25/0,35/0,50/0,65. Para beta se compara con beta0,35 fijada; los otros parámetros ID permanecen iguales. Beta0,65 no estuvo en entrenamiento y se explora por separado; no explica el shift original.

Se reutilizan los moduladores, RRC y bloques de canal validados. Cada padre reconstruye streams separados de parámetros/bits/ruido desde SeedSequence. Las versiones comparten bits, amplitud, fase, timing y ruido complejo normalizado. Se muestrean de antemano canal ID/shift y CFO ID/shift; cambiar escenario no desplaza el consumo aleatorio. El signo CFOshift conserva el del CFO ID, manteniendo marginales simétricas; la magnitud se sortea aparte.

Para evitar confundir el efecto del canal con un desplazamiento de ventana, todos los FIR se completan con ceros hasta14 taps y todas las versiones se recortan en `len(RRC)-1+13+timing`. Quedan después de los transitorios y antes del final útil de la trama. El prefijo256 es exactamente el inicio de512. El generador histórico recorta según el retardo efectivo de cada canal: esta diferencia controlada hace que el ensayo no sea una reproducción exacta del holdout.

SNR nominal: cociente entre potencia media de las muestras de la trama recibida después de canal/CFO y antes de AWGN, y potencia esperada del ruido. No es Eb/N0 ni SNR de símbolos. El ruido comparte realización estándar, pero escala según potencia de cada versión para mantener SNR nominal; no se fuerza la SNR medida de cada ventana. CFO conserva potencia, de modo que su pareja comparte también ruido absoluto. Los parámetros verdaderos y ruido oracle solo sirven al diagnóstico; inferencia recibe IQ.

Se reporta exactitud/F1 por escenario/SNR y diferencias pareadas frente a ID o beta0,35, con IC bootstrap descriptivos de10000 réplicas estratificadas por clase/SNR. Padres diferentes son independientes; versiones de un padre no lo son. Interacción factorial: accuracy(ambos)−accuracy(CFO)−accuracy(MP)+accuracy(ID), descriptiva.

Complementariedad: seis pares de modelos, conteos ambos correctos/soloA/soloB/ambos incorrectos, por escenario/SNR. El oracle de dos/tres/cuatro modelos acierta si cualquiera acierta: usa verdad y constituye solo límite superior, no política viable. No se mezclan adquisición N/fs e inferencia ni se promete ahorro a partir de este oracle. Modelos e información futura no se usan para decidir acciones reales, porque no se implementa selector.

Pruebas pertinentes: pareado de bits/ruido/crop, conservación de magnitud con CFO, aislamiento beta, reproducción determinista y padres distintos. Cada escenario guarda observaciones, metadatos oracle y predicciones por separado; NPZ mediante reemplazo atómico. Se puede reanudar saltando escenarios completos, con configuración y hashes de modelos iguales. El ensayo es exploratorio, no nueva prueba final.
