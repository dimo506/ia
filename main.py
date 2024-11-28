# Importamos las librerías necesarias
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    """
    Este programa es un sistema de aprendizaje automático simple que clasifica textos como 
    'Relevantes' o 'Irrelevantes' y mejora su precisión con el tiempo a medida que el usuario 
    proporciona retroalimentación. Todo ocurre en un único archivo.

    Pasos para ejecutar este programa:
    1. Asegúrate de tener Python instalado (https://www.python.org/).
    2. Instala la dependencia necesaria ejecutando en la terminal:
       pip install scikit-learn
    3. Guarda este archivo como 'main.py' en una carpeta.
    4. Abre una terminal, navega a la carpeta y ejecuta:
       python main.py
    5. Introduce textos para clasificar. El programa aprenderá de tu retroalimentación.
    6. Escribe 'salir' para terminar el programa.
    """

    # Inicialización del modelo de aprendizaje automático
    vectorizer = TfidfVectorizer(stop_words='english')  # Convierte texto a números
    model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)  # Clasificador incremental
    corpus = []  # Almacenará textos para entrenamiento
    labels = []  # Almacenará etiquetas: 0 = irrelevante, 1 = relevante

    # Función para entrenar el modelo con nuevos datos
    def entrenar_modelo(textos, etiquetas):
        """Actualiza el modelo con nuevos textos y etiquetas."""
        nonlocal corpus, labels  # Referenciamos las variables externas
        corpus += textos
        labels += etiquetas
        X = vectorizer.fit_transform(corpus)
        model.partial_fit(X, labels, classes=np.array([0, 1]))  # Entrena incrementalmente

    # Función para clasificar un texto
    def clasificar_texto(texto):
        """Clasifica un texto como relevante o irrelevante."""
        if len(corpus) == 0:  # Verifica si el modelo está entrenado
            return "Modelo no entrenado"
        X = vectorizer.transform([texto])  # Convierte el texto para el modelo
        prediccion = model.predict(X)[0]  # Predice la clase
        probabilidad = model.predict_proba(X)[0][prediccion]  # Confianza de la predicción
        return prediccion, probabilidad

    # Mensaje de bienvenida
    print("Bienvenido al sistema de aprendizaje continuo")
    print("Introduce textos para clasificarlos como relevantes o irrelevantes.")
    print("El sistema aprenderá a mejorar con el tiempo. Escribe 'salir' para terminar.")

    # Bucle principal
    while True:
        # Solicitar un texto para clasificar
        texto = input("\nIntroduce un texto (o escribe 'salir' para terminar): ")
        if texto.lower() == "salir":  # Condición de salida
            print("¡Gracias por usar el sistema! Hasta pronto.")
            break

        # Clasifica el texto y muestra el resultado
        resultado = clasificar_texto(texto)
        if resultado == "Modelo no entrenado":
            print("El modelo aún no está entrenado. Por favor, proporciona ejemplos.")
        else:
            prediccion, probabilidad = resultado
            etiqueta = "Relevante" if prediccion == 1 else "Irrelevante"
            print(f"Clasificación: {etiqueta} (confianza: {probabilidad:.2f})")

        # Solicita retroalimentación para entrenar el modelo
        feedback = input("¿Es relevante este texto? (1: Sí, 0: No): ")
        entrenar_modelo([texto], [int(feedback)])  # Actualiza el modelo con la entrada

# Punto de entrada del programa
if __name__ == "__main__":
    main()
