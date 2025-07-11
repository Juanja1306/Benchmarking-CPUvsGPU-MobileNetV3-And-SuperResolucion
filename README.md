# Benchmarking CPU vs GPU con MobileNetV3 y Super Resolución de Video

Este repositorio contiene implementaciones para comparar el desempeño de CPU vs GPU en distintos escenarios de visión por computadora usando OpenCV y CUDA.

## Estructura de archivos

- **Comparación de Rendimiento CPU vs GPU con MobileNetV3**: `live_classify.cpp`
  - Modelo: `mobilenetv3-small-100.onnx` (ONNX) para clasificación multilabel de ImageNet desde un stream MJPEG.
- **Comparación de Rendimiento CPU vs GPU en Super Resolución de Video**: `Principal.cpp`
  - Modelo: `LapSRN_x4.pb` (Protobuf) para superresolución x4 de vídeo usando LapSRN (Laplacian Pyramid Super-Resolution Network).
- **Desempeño Comparativo de Pipelines de Visión por Computadora en CPU y GPU usando OpenCV**:
  - `preprocess_cpu_gpu.cpp`: pipeline clásico de preprocesamiento (GaussianBlur, erode, dilate, Canny y equalizeHist) comparando rendimiento CPU vs GPU mediante GpuMat.
  - `optimized2.cpp`: benchmark de GaussianBlur (Size=15×15, sigma=3.0) en 100 iteraciones para comparar tiempos de ejecución en CPU vs GPU.

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

## Comprobar uso GPU

Para comprobar:
```bash
watch -n 1 nvidia-smi
```

Fijarce en el porcentaje(25%):
```bash
Wed Jun 25 12:51:07 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.01              Driver Version: 576.80         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce MX450           On  |   00000000:01:00.0 Off |                  N/A |
| N/A   60C    P0            N/A  / 5001W |     115MiB /   2048MiB |     25%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4835      C   /optimized                            N/A      |
+-----------------------------------------------------------------------------------------+
```

O bien:
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits -l 1
```

```bash
0, 0, 46, 2048
6, 2, 115, 2048
22, 7, 115, 2048
24, 8, 115, 2048
```


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

### Comparación de Pipelines CPU y GPU

#### Resultados Cuantitativos

| Operación           | Tiempo medio por frame |
|---------------------|------------------------|
| CPU pipeline        | 1.16884 ms             |
| GPU-only pipeline   | 8.44291 ms             |

#### Evaluación con Carga Repetida

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


# Conclusion

Este experimento demuestra la importancia de evaluar cuidadosamente el contexto antes de decidir si usar GPU. La aceleración por GPU es poderosa pero debe usarse de forma estratégica. El uso de GPU-only pipelines permite ahorrar tiempo en casos de carga elevada, aunque en tareas livianas el CPU puede superar en eficiencia por la menor sobrecarga de transferencia.