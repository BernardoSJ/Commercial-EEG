# Commercial-EEG

## Requerimientos

### MNE Python
[Instrucciones de instalación](https://mne.tools/dev/install/mne_python.html#installing-python)

### SDK de Google Cloud
[Inicio rápido](https://cloud.google.com/sdk/docs/quickstarts)

### Librerías
`tables` para lectura y escritura de archivos HDF5. Instalar con `pip`.

## Uso
Descargar datos
~~~ bash
python get_data.py
~~~

Procesar datos
~~~ bash
python preprocess_data.py
~~~

En ambos procedimientos el progreso es guardado de forma que en caso de detener el proceso por cualquier razón este continuará en dónde se había quedado la próxima vez que sea ejecutado.
