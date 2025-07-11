# Benchmarking CPU vs GPU con MobileNetV3 y Super Resolución de Video

Este repositorio contiene implementaciones para comparar el desempeño de CPU vs GPU en distintos escenarios de visión por computadora usando OpenCV y CUDA.

## Estructura de archivos

- **Comparación de Rendimiento CPU vs GPU con MobileNetV3**: `live_classify.cpp`
- **Comparación de Rendimiento CPU vs GPU en Super Resolución de Video**: `Principal.cpp`
- **Desempeño Comparativo de Pipelines de Visión por Computadora en CPU y GPU usando OpenCV**: `preprocess_cpu_gpu.cpp`, `optimized2.cpp`

## Requisitos

- OpenCV 4 con módulos de CUDA
- Compilador C++ (por ejemplo, g++)
- pkg-config
- Tarjeta Nvidia

## Compilación de OpenCV desde fuente con CMake
Para más información visitar: [OpenCV 4.11.0 con CUDA en Ubuntu](https://www.notion.so/OpenCV-4-11-0-con-CUDA-en-Ubuntu-21dae96cf28d808e9631d1836ad3214b?source=copy_link) 

```bash
cd ~/opencv
mkdir build && cd build
```

```bash
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN="7.5" \
  -D OPENCV_DNN_CUDA=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D WITH_CUBLAS=ON \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D BUILD_EXAMPLES=ON \
  -D BUILD_opencv_hdf=OFF \
  -D WITH_OPENJPEG=OFF \
  -D CMAKE_CXX_STANDARD=17 \
  ../opencv
```

## Compilación

Para compilar cualquiera de los programas, usa el siguiente comando, donde reemplaza `<archivo.cpp>` con el archivo .cpp deseado y `<ejecutable>` con el nombre que desees para el ejecutable:

```bash
g++ <archivo.cpp> -o <ejecutable> `pkg-config --cflags --libs opencv4`
```

Ejemplo para `Principal.cpp`:

```bash
g++ Principal.cpp -o Principal `pkg-config --cflags --libs opencv4`
```

## Ejecución

Para ejecutar el programa compilado, usa:

```bash
./<ejecutable>
```

Por ejemplo:

```bash
./Principal
```

### Parámetros por programa

- **`live_classify`**:  
  ```bash
  ./live_classify [-gpu | -cpu]
  ```
  - `-gpu`: habilita inferencia con CUDA.  
  - `-cpu`: fuerza inferencia por CPU (por defecto).

- **`optimized2`**:  
  ```bash
  ./optimized2 <ruta_imagen_o_video>
  ```
  - `<ruta_imagen_o_video>`: ruta al archivo de imagen o vídeo.

- **`preprocess_cpu_gpu`**:  
  ```bash
  ./preprocess_cpu_gpu [<ruta_imagen_o_video>]
  ```
  - Si no se especifica ruta, usa la cámara web por defecto.

- **`Principal`**:  
  ```bash
  ./Principal
  ```
  - No requiere parámetros (modelos y vídeo están preconfigurados).

## Resultados

### Configuración del Entorno

| Recurso            | Detalle                              |
|--------------------|--------------------------------------|
| Sistema operativo  | Ubuntu bajo WSL2                     |
| CPU                | Intel Core i7                        |
| GPU                | NVIDIA GeForce MX450 (CUDA 12.9)     |
| RAM del sistema    | 8 GB                                 |

### Comparación de Rendimiento CPU vs GPU con MobileNetV3

| Métrica                       | CPU                      | GPU                         |
|-------------------------------|--------------------------|-----------------------------|
| FPS promedio                  | 29.9 FPS                 | 29.9 FPS                    |
| Tiempo de inferencia (frame)  | 10 ms                    | 10 ms                       |
| Memoria RAM utilizada         | 1.1 GB (VmmemWSL)        | 1.2 GB (VmmemWSL)           |
| Memoria GPU utilizada         | 0                        | 75 MiB (NVIDIA MX450)       |
| Utilización GPU               | 0%                       | 1–6%                        |

### Comparación de Rendimiento CPU vs GPU en Super Resolución de Video

| Modo                         | FPS Promedio |
|------------------------------|--------------|
| CPU                          | 1.62 FPS     |
| GPU (NVIDIA GeForce MX450)   | 5.20 FPS     |

### Resultados Cuantitativos

| Operación           | Tiempo medio por frame |
|---------------------|------------------------|
| CPU pipeline        | 1.16884 ms             |
| GPU-only pipeline   | 8.44291 ms             |

### Evaluación con Carga Repetida

| Métrica                  | Valor        |
|--------------------------|--------------|
| Tiempo CPU (100 blurs)   | 1187.91 ms   |
| Tiempo GPU (100 blurs)   | 651.18 ms    |
| Aceleración (Speedup)    | 1.82x        |

### Reflexión Final

#### ¿Cuál es la diferencia entre un pipeline mixto CPU ↔ GPU y un pipeline GPU-only?

- En un pipeline mixto, los datos deben transferirse entre la memoria de host (CPU) y la memoria de dispositivo (GPU) varias veces, generando latencias innecesarias.
- En un pipeline GPU-only, todos los pasos se ejecutan en memoria GPU usando GpuMat, lo que minimiza las transferencias y mejora la eficiencia, siempre y cuando las operaciones sean suficientemente costosas para justificar el overhead.

#### ¿Cuándo vale la pena usar la GPU?

- Cuando se procesan muchos datos por lote (imágenes grandes o muchos frames).
- Cuando se realizan múltiples operaciones en cadena dentro de la GPU.
- En sistemas embebidos con soporte CUDA y necesidades de procesamiento eficiente.
- No siempre es rentable para tareas simples o imágenes pequeñas y cuando la GPU es de gama baja como en este caso.
