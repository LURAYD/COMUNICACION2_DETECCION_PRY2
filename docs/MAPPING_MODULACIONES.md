# Mapeo de las seis modulaciones, versión 1

Convención fijada para la Fase 1, 3 de noviembre de 2026. Los bits se leen de izquierda a derecha; el primero es el más significativo en cada grupo. Se preservan exactamente las tablas del código original de BPSK, 8PSK, 16-QAM, 64-QAM y 4-PAM. QPSK adopta la propuesta de ingeniería documentada en la auditoría.

## Contrato de entrada y salida

Cada función recibe un vector no vacío de bits 0/1 y devuelve un vector de símbolos complejos. Admite enteros con o sin signo, booleanos y flotantes exactamente 0.0/1.0; valida antes de convertir a entero. Rechaza matrices, escalares, valores no binarios y grupos incompletos. No agrega bits de relleno, no normaliza cada trama y no consume números aleatorios.

| Modulación | Bits por símbolo | Divisor de amplitud | Energía media de la constelación completa |
|---|---:|---:|---:|
| BPSK | 1 | 1 | 1 |
| QPSK | 2 | √2 | 1 |
| 8PSK | 3 | 1 | 1 |
| 16-QAM | 4 | √10 | 1 |
| 64-QAM | 6 | √42 | 1 |
| 4-PAM | 2 | √5 | 1 |

Estas energías corresponden a puntos equiprobables. Una trama finita QAM/PAM puede tener otra energía media: no debe corregirse cambiando su escala por trama.

## BPSK y QPSK

BPSK: `s = 1 - 2*b`: 0 → +1, 1 → −1.

QPSK: `s = ((1 - 2*b0) + j*(1 - 2*b1))/sqrt(2)`.

| Bits | I sin normalizar | Q sin normalizar |
|---|---:|---:|
| 00 | +1 | +1 |
| 01 | +1 | −1 |
| 10 | −1 | +1 |
| 11 | −1 | −1 |

La orientación de QPSK es una elección del proyecto; distintas convenciones rotadas también existen. Aquí los vecinos más próximos difieren en un bit (Gray).

## 8PSK

`s = exp(j*2*pi*k/8)`. Tabla escrita por orden angular para mostrar Gray y el cierre circular 100 → 000:

| Bits | k | Ángulo |
|---|---:|---:|
| 000 | 0 | 0° |
| 001 | 1 | 45° |
| 011 | 2 | 90° |
| 010 | 3 | 135° |
| 110 | 4 | 180° |
| 111 | 5 | 225° |
| 101 | 6 | 270° |
| 100 | 7 | 315° |

## 4-PAM y 16-QAM

| Par de bits | Nivel |
|---|---:|
| 00 | +3 |
| 01 | +1 |
| 10 | −3 |
| 11 | −1 |

4-PAM usa un par, Q=0 y divisor √5. 16-QAM usa los dos primeros bits para I y los dos últimos para Q, con divisor √10. La tabla factorizada define sin ambigüedad los 16 puntos.

La media de los niveles al cuadrado es `(9+1+9+1)/4 = 5`. Por eso 4-PAM requiere √5; las dos componentes de 16-QAM suman energía 10 antes de normalizar.

## 64-QAM

| Terna de bits | Nivel |
|---|---:|
| 000 | +7 |
| 001 | +5 |
| 010 | +1 |
| 011 | +3 |
| 100 | −7 |
| 101 | −5 |
| 110 | −1 |
| 111 | −3 |

Los tres primeros bits seleccionan I y los tres últimos Q; se divide `I+jQ` entre √42. La tabla factorizada define los 64 puntos y conserva la asignación original, incluso el orden +1/+3 y −1/−3.

Cada eje tiene energía media `(49+25+1+9+49+25+1+9)/8 = 21`; ambos ejes suman 42. Ordenados por amplitud, los vecinos difieren en un bit. Gray describe adyacencia geométrica; no exige que la lista binaria se escriba en orden creciente de amplitud.

## Qué comprueban las pruebas

Se recorren todas las palabras binarias, se comprueban puntos distintos, energía media, número de símbolos y vecinos Gray. Las pruebas de tablas exactas impiden que una rotación o permutación accidental pase inadvertida aunque conserve la energía. La reproducibilidad significa que los mismos bits producen los mismos símbolos; la semilla pertenece al generador de bits del experimento, no al modulador.
