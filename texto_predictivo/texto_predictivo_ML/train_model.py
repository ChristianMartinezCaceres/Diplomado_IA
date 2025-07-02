# entrenar_modelo.py
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import nltk
import re

try:
    nltk.download('cess_esp') # Un corpus de noticias en español del Centro de Lenguajes y Sistemas de Traducción
    print("Corpus de NLTK 'cess_esp' descargado.")
except Exception as e:
    print(f"Error al descargar corpus de NLTK: {e}")
    print("Asegúrate de tener conexión a internet.")

print("Cargando corpus desde NLTK...")

# Opción 1: Usar el corpus CESS-ESP (Corpus de la Estructura Sintáctica del Español)
try:
    from nltk.corpus import cess_esp
    all_words_from_cess = [
        re.sub(r'[^a-záéíóúüñ]', '', word.lower()) # Mantener solo letras, incluyendo acentos y 'ñ'
        for word in cess_esp.words() if word.isalpha() 
    ]
    # Eliminar posibles cadenas vacías después de la limpieza
    all_words_from_cess = [word for word in all_words_from_cess if word]
    print(f"Palabras cargadas de cess_esp: {len(all_words_from_cess)}")

except LookupError:
    print("El corpus 'cess_esp' de NLTK no está descargado.")
    print("Por favor, ejecuta 'python install_dependencies.py' y asegúrate de que 'nltk.download('cess_esp')' se ejecute.")
    all_words_from_cess = [] 


all_palabras = all_words_from_cess

# if not all_palabras:
#     print("¡Advertencia! No se pudo cargar ningún corpus de NLTK.")
#     print("Asegúrate de que los corpus estén descargados y que el nombre del corpus sea correcto.")
#     print("Usando un vocabulario de ejemplo limitado para continuar el entrenamiento.")
#     # Si por alguna razón los corpus de NLTK no se cargan,
#     # al menos entrenamos con un vocabulario pequeño para que el script funcione.
#     all_palabras = [
#     "pato", "papa", "papel", "pared", "parque", "pasto", "pelota", "perro", "persona", "pantalla",
#     "computadora", "comida", "cometa", "correr", "caminar", "carro", "casa", "cámara", "canción",
#     "abaco", "abeja", "abrir", "abrazo", "acciones", "aceptar", "acuerdo", "adelante", "adiós",
#     "agenda", "agradable", "agua", "aguila", "aire", "alegre", "alegria", "alimentos", "alma",
#     "amanecer", "amigo", "amor", "ancho", "animal", "año", "aprender", "apoyo", "aqui", "arbol",
#     "armario", "arte", "azul", "bailar", "bajo", "balon", "banco", "bandera", "barco", "basura",
#     "beber", "belleza", "biblioteca", "bicicleta", "bien", "blanco", "boca", "bola", "bosque",
#     "brazo", "bueno", "burbuja", "buscar", "caballo", "cabeza", "cabra", "cafe", "caja", "caliente",
#     "calle", "cama", "caminar", "camino", "campo", "cangrejo", "cantar", "cara", "carbon", "carne",
#     "carta", "casa", "cascada", "castillo", "cebolla", "celular", "cena", "centro", "cerebro",
#     "cerrar", "cielo", "cien", "cientifico", "circulo", "ciudad", "claro", "clima", "cocina",
#     "coco", "coche", "codigo", "coger", "cola", "coleccion", "color", "comida", "como", "compañero",
#     "comprar", "comun", "conectar", "conejo", "conocer", "consultar", "contar", "contenido",
#     "continuar", "corazon", "cordero", "corona", "correr", "corte", "cosa", "cosecha", "costa",
#     "crear", "cruz", "cuaderno", "cuadro", "cual", "cuando", "cuanto", "cuarto", "cubo", "cuchara",
#     "cuenta", "cuerpo", "cultura", "cumpleaños", "curioso", "dado", "dama", "danza", "dar", "dato",
#     "deber", "dedo", "defender", "delgado", "dentro", "derecho", "desayuno", "descansar", "deseo",
#     "desierto", "despertar", "dia", "dibujar", "diente", "dinero", "direccion", "disco", "diseño",
#     "distancia", "divertido", "doble", "doctor", "dolor", "donde", "dormir", "dos", "dragon",
#     "dulce", "duro", "echar", "edad", "edificio", "educacion", "ejemplo", "ejercicio", "el",
#     "elefante", "elegir", "elemento", "ella", "ellos", "emocion", "empezar", "empleado", "empresa",
#     "encontrar", "energia", "enfermo", "entender", "entrada", "entrar", "enviar", "escritor",
#     "escritorio", "escuela", "escribir", "escuchar", "espacio", "espalda", "especial", "esperar",
#     "esposa", "estado", "este", "estrella", "estructura", "estudiar", "evento", "exito", "experiencia",
#     "explicar", "exterior", "extraño", "facil", "familia", "famoso", "fantasma", "farmacia",
#     "favor", "feliz", "feminino", "fiesta", "figura", "fin", "final", "flor", "fluido", "fondo",
#     "forma", "formar", "fotografia", "frio", "fruta", "fuego", "fuerza", "funcion", "futuro",
#     "galleta", "ganar", "gato", "generacion", "gente", "gigante", "girar", "global", "golpe",
#     "grande", "gratis", "grave", "gritar", "grupo", "guardar", "guerra", "gustar", "haber", "habitacion",
#     "hablar", "hacer", "hacia", "hallar", "hermana", "hermano", "herramienta", "hielo", "hierro",
#     "historia", "hoja", "hombre", "hombro", "hora", "hospital", "hoy", "hueso", "humano", "idea",
#     "iglesia", "imagen", "importante", "imposible", "impresora", "incluir", "indicar", "individuo",
#     "informacion", "inicio", "insecto", "insertar", "interior", "interes", "internacional",
#     "internet", "invertir", "invierno", "ir", "isla", "izquierda", "jardin", "jefe", "joven",
#     "juego", "jugar", "junto", "justo", "kilometro", "labor", "lado", "lago", "lapiz", "largo",
#     "lavar", "leche", "leer", "legal", "lengua", "lento", "leon", "letra", "levantar", "ley",
#     "liberar", "libro", "lider", "limpiar", "linea", "lista", "litro", "llamar", "llave", "llegar",
#     "llenar", "llevar", "llorar", "llover", "local", "lobo", "lograr", "londres", "loro", "luchar",
#     "lugar", "luna", "luz", "madera", "madre", "maestra", "maestro", "magia", "mal", "malo",
#     "mama", "manana", "mano", "mantener", "mapa", "mar", "mariposa", "mas", "masa", "matar",
#     "material", "mayo", "mayor", "medida", "medio", "mejor", "memoria", "mensaje", "mente",
#     "mesa", "meter", "metro", "mezclar", "miedo", "mientras", "miercoles", "mil", "mirar",
#     "mismo", "mitad", "modelo", "moderno", "modo", "mojar", "momento", "monitor", "mono", "montaña",
#     "moral", "mostrar", "mover", "mucho", "mueble", "muerte", "mujer", "mundo", "musica",
#     "nacer", "nacional", "nada", "nadar", "naranja", "nariz", "naturaleza", "navegar", "necesitar",
#     "negocio", "negro", "nervioso", "nieve", "niña", "niño", "nivel", "noche", "nombre", "normal",
#     "norte", "noticia", "nube", "nuestro", "nuevo", "numero", "nunca", "objeto", "obra", "observar",
#     "obtener", "oceano", "ocho", "ocupado", "oeste", "oficina", "oir", "ojo", "ola", "oler",
#     "olvidar", "once", "operacion", "opinion", "oportunidad", "orden", "organizar", "origen",
#     "oro", "oscuro", "otoño", "otro", "oveja", "paciente", "padre", "pago", "pais", "pajaro",
#     "palabra", "palma", "pan", "pantalon", "papa", "papel", "par", "para", "parar", "parecer",
#     "pared", "pareja", "parque", "parte", "participar", "particular", "pasado", "pasar", "paso",
#     "pata", "patio", "paz", "pecho", "pedir", "pelicula", "pelo", "pelota", "pena", "pensar",
#     "pequeño", "perder", "perfecto", "periodo", "permitir", "perro", "persona", "pesar", "pescado",
#     "peso", "pez", "pie", "piedra", "piel", "pierna", "pieza", "pintar", "pintura", "piña",
#     "piojo", "pisar", "piscina", "placer", "plan", "planta", "plata", "playa", "pluma", "pobre",
#     "poco", "poder", "policia", "politica", "polvo", "poner", "por", "porque", "posible", "posicion",
#     "practica", "precio", "preciso", "preferir", "pregunta", "preparar", "presentar", "presidente",
#     "presion", "prestar", "primer", "primo", "principal", "principio", "privado", "problema",
#     "proceso", "producir", "profesor", "programa", "proyecto", "prueba", "publico", "pueblo",
#     "puente", "puerta", "punto", "puro", "queso", "quedar", "querer", "quien", "quimica",
#     "quince", "quitar", "radio", "raiz", "ramo", "rapido", "rato", "razon", "real", "realidad",
#     "recibir", "recuerdo", "redondo", "referir", "region", "regresar", "relacion", "reloj",
#     "representar", "respuesta", "restaurante", "resultado", "reunion", "revista", "rey", "rico",
#     "riesgo", "rio", "riqueza", "rodilla", "rojo", "romper", "ropa", "rosa", "rubio", "ruido",
#     "rusa", "saber", "sacar", "sabado", "sabor", "sacrificio", "salir", "salud", "salvar", "sangre",
#     "santa", "santo", "sara", "seguir", "segundo", "seguro", "seis", "selva", "semana", "sembrar",
#     "sentar", "sentido", "señal", "separar", "ser", "serie", "servicio", "setiembre", "sexo",
#     "siempre", "siete", "significar", "silencio", "silla", "simple", "sin", "sino", "sistema",
#     "sitio", "situacion", "social", "sol", "solo", "solucion", "sombra", "sombrero", "sonar",
#     "soñar", "sonido", "sonrisa", "sopa", "sorpresa", "suave", "subir", "sucio", "suelo", "suerte",
#     "suficiente", "sugerir", "suma", "superficie", "suponer", "sur", "sus", "tal", "tambien",
#     "tan", "tanto", "tarde", "tarea", "te", "teatro", "techo", "teclado", "tecnologia", "telefono",
#     "television", "tema", "temprano", "tener", "terminar", "terrible", "texto", "tia", "tiempo",
#     "tienda", "tierra", "tigre", "tipo", "tocar", "todavia", "todo", "tomar", "tono", "tonto",
#     "tormenta", "total", "trabajo", "tradicion", "traer", "transformar", "transporte", "tratar",
#     "trece", "tren", "tres", "triangulo", "triste", "trompeta", "tu", "tunel", "ultimo", "uno",
#     "urgente", "usar", "uso", "usted", "util", "vaca", "vacaciones", "valle", "valor", "vapor",
#     "varios", "vecino", "veinte", "velocidad", "vender", "venir", "ventana", "ver", "verano",
#     "verdad", "verde", "vestido", "vez", "viaje", "vida", "viejo", "viento", "viernes", "violencia",
#     "virgen", "virtud", "visible", "vision", "visitar", "vista", "vivir", "volar", "volver",
#     "volumen", "voz", "vuelta", "y", "ya", "yegua", "yo", "zona", "zorro"
#     ]

all_palabras_unique = list(set(all_palabras))
# Opcional: Filtrar palabras muy cortas (ej. de 1 o 2 letras) si no son relevantes
all_palabras_unique = [word for word in all_palabras_unique if len(word) > 1]
print(f"Número de palabras únicas y limpias en el vocabulario final: {len(all_palabras_unique)}")

df = pd.DataFrame(all_palabras_unique, columns=["palabra"])

# Vectorizamos las palabras
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(df["palabra"])
y = df["palabra"]

# Entrenamos un modelo de clasificación 
model = LogisticRegression()
model.fit(X, y)

# Guardamos modelo y vectorizador
joblib.dump(model, "modelo_predictivo.pkl")
joblib.dump(vectorizer, "vectorizador.pkl")
