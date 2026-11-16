# Referencia externa incorporada en Fase 1

`SquareRootRaisedCosine.pdf` se descargó el 3 de noviembre de 2026 desde [Purdue, Equations for the Raised Cosine and Square-Root Raised Cosine Shapes](https://engineering.purdue.edu/~ee538/SquareRootRaisedCosine.pdf). No forma parte del corpus original de 17 PDF auditado en Fase 0.

SHA-256: `3dc327a897ec046e0e581acc2182d33e56a968f553dcf4c59b24ec594cb180ed`.

Se inspeccionaron visualmente las páginas PDF/impresas 1, 3 y 5. La página 3 contiene una diferencia de escala respecto de la fórmula usada; véase `docs/MODELO_DE_SENAL.md`. Se usa el espectro RC de la p. 1 para el contraste espectral y la sección de truncamiento de la p. 5 como contexto, no como fuente de los límites singulares.

Fuentes web adicionales consultadas, sin instalar sus bibliotecas ni ejecutar sus programas:

- [OptiCommPy, rrcFilterTaps, ecuaciones (1)–(3)](https://opticommpy.readthedocs.io/en/stable/_modules/optic/dsp/core.html#rrcFilterTaps): fórmula general y límites.
- [MathWorks, rcosdesign](https://www.mathworks.com/help/signal/ref/rcosdesign.html): energía discreta, longitud y cascada de filtros finitos.
