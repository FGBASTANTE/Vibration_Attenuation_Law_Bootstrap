# Vibration_Attenuation_Law_Bootstrap
Aplicación del método de bootstrap a la determinación de la recta(curva) de 
seguridad de la ley de propagación de vibraciones en el terreno
Se espera que se haya definido previamente el modelo de distancia escalada
(s_d: Distancia/Carga^beta). Como ejemplo, y típicamente para cargas alargadas:
beta = 1/2, y para cargas esféricas: beta = 1/3.

En el fichero de entrada los valores x son los logaritmos decimales de
las distancias escaladas (log10(s_d)); los valores y son, consecuentemente,
los log10(ppv):\
x	y\
1.76779	0.2001\
0.69139	1.96096\
1.55308	1.06786\
..............\

También está implementado el modelo lognormal para comparar resultados.

![image](https://github.com/user-attachments/assets/4d34222b-bc32-41ae-8e27-6d91cdb5a268)

