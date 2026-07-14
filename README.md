Primero, instala Python para ejecutar los archivos
build_dataset toma un archivo de Excel para convertirlo en un parquet usando funciones de data_processing_utils
data_processing_utils tiene funciones para preocesar los datos
lpsmltest tiene functiones para enseñar al modelo a predicir las tarifas
lpsmltraining usa los datos para enseñar al modelo
MAE.png muestra puntos demonstrando la comparacion entre el valor predicida y actual de la tarifa
Hist_error.png muestra la distribucion del error del modelo
Hist_error_zoomed.png muestra la distribucion del 90% de datos para ver con mas detalle
Usa build_dataset y despues lpsmltraining
