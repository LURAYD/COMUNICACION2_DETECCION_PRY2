# Modelo de señal validado en Fase 1

## Símbolos y muestras

El modulador agrupa k bits y obtiene símbolos complejos `a[m]`. Con `sps` muestras por símbolo, la inserción de ceros forma `u[m*sps]=a[m]`; los demás valores de u son cero. La señal de transmisión es `x[n] = sum_m a[m]*h[n-m*sps]`.

`sps` es un número de muestras por símbolo, no una frecuencia. Si se conoce la tasa de símbolos `Rs` en baud, `fs = sps*Rs` en muestras/s y `Ts=1/Rs`. Los diagnósticos usan tiempo en períodos de símbolo y frecuencia `f/Rs`; no suponen un `fs` físico.

## RRC, fórmula y fuentes externas

El libro de Medina sigue ausente. La ecuación se verificó como conocimiento externo contra la documentación del propio proyecto [OptiCommPy, rrcFilterTaps, ecuaciones (1)–(3)](https://opticommpy.readthedocs.io/en/stable/_modules/optic/dsp/core.html#rrcFilterTaps), consultada el 3 de noviembre de 2026. Se tomó `Ts=1` para la forma adimensional; no se instaló ni se ejecutó OptiCommPy.

Para `tau=t/Ts`, antes de normalizar:

```text
p(tau) = [sin(pi*tau*(1-beta)) + 4*beta*tau*cos(pi*tau*(1+beta))]
         / [pi*tau*(1-(4*beta*tau)^2)]

p(0) = 1 + beta*(4/pi - 1)

p(±1/(4*beta)) = beta/sqrt(2) * [
    (1+2/pi)*sin(pi/(4*beta)) + (1-2/pi)*cos(pi/(4*beta))
]
```

Los casos especiales son límites finitos de una expresión con división 0/0; no son valores arbitrarios de reemplazo. Se muestrea `tau=-span/2,...,+span/2` con paso `1/sps`, y se normaliza `h = p/sqrt(sum(p*p))`. La energía discreta es uno. El filtro conserva el dominio del código original: `0 < beta <= 1`, `sps` entero ≥2 y `span` entero positivo par. No se implementa beta=0.

La documentación de [MathWorks, rcosdesign](https://www.mathworks.com/help/signal/ref/rcosdesign.html), secciones Description y Raised Cosine FIR Filters, confirma la longitud `span*sps+1`, la energía unitaria y la aproximación RC al poner dos RRC finitos en cascada. Es una verificación documental, no una comparación ejecutada en MATLAB.

También se inspeccionó visualmente [Equations for the Raised Cosine and Square-Root Raised Cosine Shapes, Purdue](https://engineering.purdue.edu/~ee538/SquareRootRaisedCosine.pdf), páginas PDF/impresas 1, 3 y 5. La ecuación (1), p. 1, define el espectro RC utilizado como referencia independiente; la sección 3, p. 5, explica truncamiento. **Discrepancia registrada:** la ecuación (5), p. 3, muestra un prefactor `2*beta/(pi*sqrt(Ts))`, la mitad del que resulta de la fórmula usada con `Ts=1`. No se adopta esa escala ni se atribuyen a ese PDF los límites singulares. La normalización energética eliminaría un factor global positivo, pero no justifica omitir la discrepancia. La fórmula y sus límites se contrastaron con OptiCommPy.

## Longitud y retardo

Con N símbolos, el código conserva el vector de inserción de ceros de longitud `N*sps`, incluyendo `sps-1` ceros tras el último símbolo. Por ello la convolución completa produce:

```text
L = span*sps + 1
len(x) = N*sps + L - 1
retardo de transmisión = (L-1)/2 muestras
retardo de transmisión + filtro adaptado = L-1 muestras
```

El filtro adaptado usa `conj(h[::-1])`. En el caso RRC real y simétrico coincide con h. Para `sps=8, span=10`, L=81, retardo individual=40 muestras (5 símbolos) y acumulado=80 (10 símbolos).

En un diagnóstico sin canal, después del filtro adaptado se observan las posiciones `L-1 + arange(N)*sps`. El pico de la respuesta combinada es 1 porque es la energía de h. Los valores restantes a intervalos de sps no son exactamente cero: el RRC está truncado. No se recorta ni compensa retardo implícitamente en `pulse_shape`; el experimento selecciona los instantes explícitos para explicar la recuperación.

## Contraste espectral independiente

Para `nu=abs(f/Rs)`, el RC ideal de pico unitario vale:

```text
1                                            si nu <= (1-beta)/2
0.5*[1+cos(pi/beta*(nu-(1-beta)/2))]           en la transición
0                                            si nu >= (1+beta)/2
```

Se compara con `abs(FFT(h))**2/sps`: la división por sps corresponde a la escala ideal de un pulso de energía discreta unitaria. La ganancia de continua del FIR truncado puede diferir ligeramente de uno; no se reajusta para ocultar esa diferencia. Esta escala es solo para el diagnóstico espectral y no cambia h ni la señal transmitida. Las ondulaciones y la energía fuera de banda son consecuencias del truncamiento, por lo que no se exige identidad exacta.

## Alcance físico

La Fase 1 valida el generador en banda base y su sincronización ideal. El filtro adaptado se utiliza como evidencia de este generador, no como receptor que ya conozca el canal. AWGN existente conserva sus pruebas de regresión de Fase 0. CFO, fase, multipath y ADC pertenecen a la Fase 2 y no se añaden aquí. No hay todavía datos de entrenamiento, clasificador ni medición de desempeño de reconocimiento.
