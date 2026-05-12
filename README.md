# DocBase — Documentación Técnica

Sistema de recuperación de información sobre legislación laboral mexicana
basado en Indexación Semántica Latente (LSI), MySQL y Streamlit.

**Curso:** LIS-3012 Bases de Datos Avanzadas — UDLAP  
**Profesor:** Dr. José Luis Zechinelli Martini

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Requisitos del sistema](#2-requisitos-del-sistema)
3. [Instalación](#3-instalación)
4. [Configuración](#4-configuración)
5. [Corpus de documentos](#5-corpus-de-documentos)
6. [Uso de la aplicación](#6-uso-de-la-aplicación)
7. [Arquitectura del sistema](#7-arquitectura-del-sistema)
8. [Módulos del backend](#8-módulos-del-backend)
9. [Esquema de base de datos](#9-esquema-de-base-datos)
10. [Pipeline de ingesta](#10-pipeline-de-ingesta)
11. [Motor de consulta y funciones de similitud](#11-motor-de-consulta-y-funciones-de-similitud)
12. [Capa LLM](#12-capa-llm)
13. [Interfaz Streamlit](#13-interfaz-streamlit)
14. [Suite de pruebas](#14-suite-de-pruebas)
15. [Referencia de comandos](#15-referencia-de-comandos)
16. [Decisiones de diseño y limitaciones conocidas](#16-decisiones-de-diseño-y-limitaciones-conocidas)
17. [Solución de problemas](#17-solución-de-problemas)

---

## 1. Visión general

DocBase indexa un corpus de documentos de legislación laboral mexicana
y permite consultarlos mediante lenguaje natural. El sistema:

- Extrae texto de PDFs (incluyendo PDFs cifrados/indexados)
- Preprocesa el texto en español con lista de palabras vacías, sufijos
  y raíces léxicas almacenados en MySQL
- Construye una matriz de frecuencias TF-IDF (FrecT) persistida en la
  tabla `HAS` como triples (documento, término, peso)
- Aplica Descomposición en Valores Singulares (SVD) con `scipy` para
  reducir el espacio semántico a k dimensiones latentes
- Ejecuta cinco funciones de similitud/distancia como consultas SQL
  puras sobre MySQL
- Presenta resultados en una interfaz Streamlit con fragmentos
  relevantes y descarga de documentos
- Opcionalmente sintetiza una respuesta en lenguaje natural usando un
  LLM local (Ollama) o la API de Gemini

**Corpus actual:** 10+ documentos de legislación laboral mexicana
incluyendo Artículo 123 constitucional, LFT, LFTSE, LFPED, LIFNVT,
LISR, LSAR, LSS, NOM-035, NOM-036, NOM-037 y otros.

**Métricas del corpus indexado:**
- Términos únicos: ~16,878
- Celdas no nulas en FrecT: ~49,600
- Rango SVD k: 9 (máximo para 10 documentos)
- Tiempo de consulta promedio: ~7ms
- Rango de similitud coseno entre documentos: 0.06 – 0.71

---

## 2. Requisitos del sistema

| Componente | Versión mínima |
|---|---|
| Python | 3.9+ |
| MySQL | 8.x |
| RAM | 4 GB (8 GB recomendado para LLM local) |
| Sistema operativo | Linux, macOS, Windows |
| Ollama | Opcional, para LLM local |

---

## 3. Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/docbase.git
cd docbase

# 2. Crear y activar entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear base de datos MySQL
mysql -u tu_usuario -p < schema.sql

# 5. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales (ver sección 4)

# 6. Colocar PDFs en data/pdfs/ (ver sección 5)

# 7. Ejecutar ingesta
python ingest.py

# 8. Lanzar la aplicación
streamlit run app.py
# Abrir: http://localhost:8501
```

---

## 4. Configuración

Editar el archivo `.env` en la raíz del proyecto:

```env
# Base de datos MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USER=tu_usuario_mysql
DB_PASS=tu_contraseña_mysql
DB_NAME=docbase

# LLM (opcionales)
GEMINI_API_KEY=tu_clave_gemini   # obtener en aistudio.google.com
OLLAMA_MODEL=llama3.2:3b          # modelo por defecto si no se detecta
```

El archivo `.env` está excluido de git por `.gitignore`. Nunca
lo compartas ni lo subas al repositorio.

### Variables de entorno

| Variable | Requerida | Descripción |
|---|---|---|
| `DB_HOST` | Sí | Host del servidor MySQL |
| `DB_PORT` | Sí | Puerto MySQL (default 3306) |
| `DB_USER` | Sí | Usuario MySQL con permisos sobre `docbase` |
| `DB_PASS` | Sí | Contraseña del usuario MySQL |
| `DB_NAME` | Sí | Nombre de la base de datos (default `docbase`) |
| `GEMINI_API_KEY` | No | Clave de API de Google Gemini (fallback LLM) |
| `OLLAMA_MODEL` | No | Modelo Ollama por defecto |

---

## 5. Corpus de documentos

Colocar los PDFs en `data/pdfs/`. El sistema acepta cualquier PDF
legible por `pdfplumber`, incluyendo PDFs cifrados con contraseña vacía.

### Documentos actualmente indexados

| Archivo | Documento |
|---|---|
| `Artículo_123.pdf` | Artículo 123 Constitucional — Derechos laborales |
| `LFT.pdf` | Ley Federal del Trabajo |
| `LFTSE.pdf` | Ley Federal de Trabajadores al Servicio del Estado |
| `LFPED.pdf` | Ley Federal para Prevenir y Eliminar la Discriminación |
| `LIFNVT.pdf` | Ley del Instituto del Fondo Nacional de la Vivienda |
| `LISR.pdf` | Ley del Impuesto sobre la Renta |
| `LSAR.pdf` | Ley del Sistema de Ahorro para el Retiro |
| `LSS.pdf` | Ley del Seguro Social |
| `NOM035.pdf` | NOM-035-STPS-2018 — Riesgo psicosocial |
| `NOM036.pdf` | NOM-036-STPS-2018 — Factores ergonómicos |
| `NOM037.pdf` | NOM-037-STPS-2023 — Teletrabajo |

### Agregar documentos al corpus

```bash
# Copiar nuevo PDF
cp /ruta/al/nuevo_documento.pdf data/pdfs/

# Re-indexar solo el nuevo documento (idempotente)
python ingest.py

# Si se quiere re-indexar todo desde cero
python ingest.py --reset
```

### Reemplazar el corpus completo

```bash
# Limpiar carpeta y agregar nuevos PDFs
rm data/pdfs/*.pdf
cp /ruta/nuevos/*.pdf data/pdfs/

# Re-indexar desde cero
python ingest.py --reset
```

---

## 6. Uso de la aplicación

### Iniciar la aplicación

```bash
source .venv/bin/activate
streamlit run app.py
```

La aplicación abre automáticamente en `http://localhost:8501`.

### Tab: Consulta

Campo de texto para escribir una consulta en lenguaje natural.
Presionar **Enter** o el botón **Buscar** inicia la búsqueda.

**Recomendaciones para mejores resultados:**
- Usar terminología jurídica en lugar de lenguaje coloquial
- Incluir 3-5 términos específicos del dominio
- Evitar preguntas completas — preferir frases nominales

| En lugar de... | Usar... |
|---|---|
| ¿Qué pasa si me despiden? | `despido injustificado rescisión indemnización` |
| ¿Cuántas horas puedo trabajar? | `jornada máxima horas diurna nocturna` |
| ¿Qué es el IMSS? | `seguridad social IMSS cuotas obligaciones patrón` |

**Controles del sidebar:**

| Control | Opciones | Descripción |
|---|---|---|
| Función de similitud | coseno, Dice, Jaccard, euclidiana, Manhattan | Método de comparación |
| Documentos a mostrar | 1–10 | Top-N resultados |
| Rango SVD (k) | 1–9 | Dimensiones del espacio latente |
| Modelo de lenguaje | Lista de modelos disponibles / Desactivado | LLM para síntesis |

**Interpretar resultados:**
- El score coseno está en [0, 1]; valores > 0.5 indican alta relevancia
- "Términos coincidentes" muestra cuántos tokens de la consulta aparecen en el documento
- El fragmento relevante es el párrafo con mayor densidad de términos de la consulta
- El botón "Descargar" descarga el PDF original del documento

**Síntesis LLM:**
Aparece debajo de los resultados cuando hay un modelo disponible.
Incluye una sección "Fuente principal:" con cita textual del fragmento
más relevante del documento con mayor score.

### Tab: Matriz FrecT

Visualiza la matriz de frecuencias TF-IDF como tabla interactiva
y mapa de calor. Usar el slider para filtrar los N términos con
mayor peso total. Útil para verificar que la ingesta funcionó
correctamente — los términos con mayor peso deben ser términos
jurídicos específicos, no palabras vacías.

### Tab: Explorador SVD

Dos gráficas de la descomposición SVD:
- **Valores singulares:** energía de cada componente k
- **Varianza explicada acumulada:** cuánta información retiene cada k

Regla práctica: k óptimo es donde la curva de varianza acumulada
supera el 80%. La línea de referencia gris lo marca.

También muestra la matriz de similitud coseno entre todos los
documentos del corpus como mapa de calor simétrico.

### Tab: Documentos

Navegador del corpus completo. Cada documento expandible muestra:
- Metadatos (autor, fecha)
- Fragmento de texto extraído
- Estadísticas: términos únicos, peso total, peso máximo
- Top 5 términos por peso TF-IDF

---

## 7. Arquitectura del sistema

```
data/pdfs/          SQL queries         localhost:8501
    |                    |                    |
    v                    v                    v
pdf_extractor     similarity.py          app.py
    |                    |                    |
    v                    v                    |
preprocessor      query_engine  <------------+
    |                    |
    v                    v
matrix_builder    svd_engine
    |                    |
    +--------------------+
             |
             v
         MySQL 8
        (docbase)
             |
             v
          llm.py
     (Ollama / Gemini)
```

### Dos fases de operación

**Ingesta (ejecuta una vez):**
PDF → extracción → preprocesamiento → FrecT → SVD → MySQL

**Consulta (ejecuta en cada búsqueda):**
Texto → preprocesamiento → vector LSI → SQL similitud → top-N → LLM

---

## 8. Módulos del backend

### `src/db.py`
Gestión de conexiones MySQL.

```python
from src.db import get_db

with get_db() as conn:
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM DOCUMENT")
    docs = cursor.fetchall()
```

Funciones principales:
- `get_connection()` — conexión raw con autocommit=False
- `execute_query(conn, sql, params)` — SELECT, retorna lista de dicts
- `execute_write(conn, sql, params)` — INSERT/UPDATE/DELETE
- `execute_many(conn, sql, rows)` — batch insert, lotes de 500 filas
- `get_db()` — context manager con rollback automático en excepción

### `src/pdf_extractor.py`
Extracción de texto de PDFs.

```python
from src.pdf_extractor import extract_text, extract_all

# Un PDF
text = extract_text("data/pdfs/LFT.pdf")

# Todo el corpus
docs = extract_all("data/pdfs")
# retorna: {'LFT': 'texto...', 'NOM035': 'texto...', ...}
```

Funciones:
- `extract_text(pdf_path)` — extrae texto de un PDF; reintenta con password='' si está cifrado; elimina cabeceras/pies de página heurísticamente
- `extract_all(pdf_dir)` — itera el directorio, omite PDFs que fallan con WARNING

### `src/preprocessor.py`
Pipeline NLP completo en español.

```python
from src.preprocessor import (load_stop_words, load_suffix_rules,
                                preprocess, preprocess_query)

with get_db() as conn:
    sw = load_stop_words(conn)   # set de palabras vacías
    sr = load_suffix_rules(conn) # lista de (sufijo, reemplazo)

tokens = preprocess(text, sw, sr)
# Mismo pipeline para consultas:
q_tokens = preprocess_query("jornada laboral", sw, sr)
```

Pipeline completo:
1. Minúsculas + normalización
2. Tokenización por regex `[^a-záéíóúüñ]+`
3. Filtro de palabras vacías (tabla `STOP_WORD`)
4. Eliminación de sufijos ordenados de mayor a menor longitud
   (tabla `SUFFIX`), solo si el tallo resultante tiene ≥ 3 chars
5. Segundo filtro de palabras vacías post-stemming
6. Filtro de longitud mínima: tokens < 3 chars descartados

### `src/matrix_builder.py`
Construcción de la matriz FrecT con TF-IDF.

```python
from src.matrix_builder import (build_frequency_matrix,
                                  apply_tfidf, save_terms, save_matrix)

matrix, term_index, doc_index = build_frequency_matrix(preprocessed_docs)
weighted = apply_tfidf(matrix, term_index, doc_index)
term_db_ids = save_terms(conn, term_index)
rows = save_matrix(conn, weighted, term_index, doc_index,
                   term_db_ids, doc_db_ids)
```

La fórmula TF-IDF usada:
```
TF(i,j)  = freq(i,j) / total_terms(j)
IDF(i)   = max(0, log(N / (1 + df(i))))
FrecT(i,j) = TF(i,j) × IDF(i)
```

El `max(0,·)` evita pesos negativos cuando `df ≥ N`.

### `src/svd_engine.py`
Descomposición SVD y proyección de consultas.

```python
from src.svd_engine import (compute_svd, save_svd,
                              get_document_vectors, project_query)

U, s, Vt = compute_svd(matrix, k=9)
save_svd(conn, U, s, Vt, term_index, doc_index,
         term_db_ids, doc_db_ids, k=9)

# En tiempo de consulta:
doc_vectors = get_document_vectors(conn, k=9)
q_vec = project_query(tokens, term_index, U, s)
```

Notas de implementación:
- Usa `scipy.sparse.linalg.svds` (versión dispersa, más eficiente)
- k se clampea automáticamente a `min(shape) - 1`
- Vector de inicio `v0` determinístico con `seed=42` para evitar
  el error ARPACK -9 en matrices con columnas casi nulas
- Valores singulares reordenados de mayor a menor (svds devuelve
  orden ascendente)

### `src/similarity.py`
Cinco funciones de similitud/distancia como SQL puro.

```python
from src.similarity import (cosine_similarity, dice_similarity,
                              jaccard_similarity, euclidean_distance,
                              manhattan_distance)

# Consulta contra todos los documentos
results = cosine_similarity(conn, query_vector=q_vec, top_n=5)
# [{'document_id': 3, 'title': 'LFT', 'score': 0.7234}, ...]

# Comparar dos documentos específicos
results = cosine_similarity(conn, query_doc_id=1, top_n=10)
```

Todas las funciones aceptan:
- `query_doc_id` — comparar un documento del corpus contra los demás
- `query_vector` — dict `{term_id: frequency}` para consultas nuevas
- `top_n` — número de resultados a retornar

Para `query_vector`, se usa una tabla temporal `QUERY_VECTOR` de
sesión, evitando tablas permanentes.

**SQL de referencia — similitud coseno:**
```sql
SELECT b.document_id,
       SUM(a.frequency * b.frequency) /
       (SQRT(SUM(a.frequency * a.frequency)) *
        SQRT(SUM(b.frequency * b.frequency))) AS score
FROM QUERY_VECTOR a
JOIN HAS b ON a.term_id = b.term_id
GROUP BY b.document_id
ORDER BY score DESC
LIMIT :top_n
```

### `src/query_engine.py`
Orquestador del pipeline completo de consulta.

```python
from src.query_engine import query, compare_documents, all_similarities

# Consulta de texto libre
results = query("jornada laboral salario", conn,
                method='cosine', top_n=5, k=9)

# Comparar dos documentos
comp = compare_documents(1, 3, conn, method='cosine')
# {'doc_a': ..., 'doc_b': ..., 'score': 0.71, 'interpretation': 'similar'}

# Todos los métodos sobre un documento
all_r = all_similarities(doc_id=1, conn, top_n=3)
```

Caché a nivel de módulo: `_cache = {'stop_words': None, 'suffix_rules': None}`.
Las reglas se cargan de MySQL una sola vez por sesión y se reutilizan
en consultas subsecuentes.

### `src/llm.py`
Capa LLM con detección automática y degradación elegante.

```python
from src.llm import detect_llm, get_available_models, synthesize,
                     get_document_text

# Detectar mejor modelo disponible
info = detect_llm()
# {'provider': 'ollama', 'model': 'gemma2:2b', 'available': True}

# Listar todos los modelos para selector manual
models = get_available_models()
# {'options': ['Desactivado', 'Local — gemma2:2b', 'Gemini 1.5 Flash'],
#  'values': [('none',None), ('ollama','gemma2:2b'), ('gemini','gemini-1.5-flash')],
#  'default_index': 1}

# Sintetizar respuesta
answer = synthesize(
    query="jornada máxima",
    results=top_n_results,
    provider='ollama',
    model='gemma2:2b',
    query_tokens=['jornada', 'maxim']
)

# Obtener fragmento más relevante
passage = get_document_text("data/pdfs/LFT.pdf",
                             query_tokens=['jornada', 'maxim'],
                             max_chars=800)
```

**Selección automática de modelo Ollama por RAM disponible:**

| RAM disponible | Modelo seleccionado |
|---|---|
| ≥ 15 GB | `qwen2.5:7b` (mejor español) |
| ≥ 7 GB | `llama3.2:3b` |
| < 7 GB | `phi3:mini` |

**Prompt del sistema:**
El LLM recibe instrucciones en español para responder únicamente
basándose en los fragmentos proporcionados e incluir al final una
sección "Fuente principal:" con cita textual del documento más relevante
(marcado como `[DOCUMENTO MÁS RELEVANTE]` en el prompt).

---

## 9. Esquema de base de datos

Base de datos: `docbase` con `utf8mb4_spanish_ci` (manejo correcto
de á, é, í, ó, ú, ü, ñ y comparaciones insensibles a mayúsculas).

### Diagrama de relaciones

```
DOCUMENT (id, url, title, author, doc_date)
    |
    |---(1,n)--- HAS (document_id FK, term_id FK, frequency)
    |                           |
    |                      (1,n)|
    |                     TERM (id, name UNIQUE)
    |                           |
    |                      (1,n)|
    |                     WORD (id, term_id FK, word)
    |
    |---(1,n)--- QUERY (id, label, document_id FK)
    |
    |---(1,n)--- SVD_MATRIX (id, term_id FK, document_id FK,
                              t_value, s_value, d_value, k_rank)

STOP_WORD (id, word UNIQUE)
SUFFIX    (id, suffix, replacement)
```

### Descripción de tablas

**DOCUMENT** — Metadatos de cada PDF del corpus
| Columna | Tipo | Descripción |
|---|---|---|
| id | INT PK AUTO_INCREMENT | Identificador único |
| url | VARCHAR(512) | Nombre del archivo PDF |
| title | VARCHAR(255) | Título del documento |
| author | VARCHAR(255) | Autor/organismo emisor |
| doc_date | DATE | Fecha del documento |

**TERM** — Vocabulario tras preprocesamiento
| Columna | Tipo | Descripción |
|---|---|---|
| id | INT PK AUTO_INCREMENT | Identificador único |
| name | VARCHAR(100) UNIQUE | Forma stemmed del término |

**WORD** — Formas originales que mapean a cada término
| Columna | Tipo | Descripción |
|---|---|---|
| id | INT PK AUTO_INCREMENT | Identificador único |
| term_id | INT FK → TERM | Término padre |
| word | VARCHAR(100) | Forma original de la palabra |

**HAS** — La matriz FrecT en formato disperso (núcleo del sistema)
| Columna | Tipo | Descripción |
|---|---|---|
| document_id | INT PK FK → DOCUMENT | Documento |
| term_id | INT PK FK → TERM | Término |
| frequency | FLOAT | Peso TF-IDF de este término en este documento |

*Clave primaria compuesta garantiza unicidad e idempotencia.*

**SVD_MATRIX** — Descomposición T, S, D almacenada por componente
| Columna | Tipo | Descripción |
|---|---|---|
| id | INT PK AUTO_INCREMENT | Identificador único |
| term_id | INT FK → TERM | Término |
| document_id | INT FK → DOCUMENT | Documento |
| t_value | FLOAT | Componente de U (matriz de términos) |
| s_value | FLOAT | Valor singular |
| d_value | FLOAT | Componente de Vt (matriz de documentos) |
| k_rank | INT | Índice del componente (0 a k-1) |

**STOP_WORD** — Lista de palabras vacías para preprocesamiento
| Columna | Tipo | Descripción |
|---|---|---|
| id | INT PK AUTO_INCREMENT | Identificador único |
| word | VARCHAR(100) UNIQUE | Palabra vacía en minúsculas |

**SUFFIX** — Reglas de eliminación de sufijos
| Columna | Tipo | Descripción |
|---|---|---|
| id | INT PK AUTO_INCREMENT | Identificador único |
| suffix | VARCHAR(50) | Sufijo a eliminar |
| replacement | VARCHAR(50) | Reemplazo (vacío = eliminar) |

### Vistas SQL creadas

```sql
-- Similitud coseno entre todos los pares de documentos
SELECT * FROM v_cosine_similarity;

-- Distancia euclidiana entre todos los pares
SELECT * FROM v_euclidean_distance;

-- Norma L2 y L1 de cada documento con conteo de términos
SELECT * FROM v_document_norms;
```

### Índices

```sql
CREATE INDEX idx_has_term ON HAS(term_id);
CREATE INDEX idx_has_doc  ON HAS(document_id);
```

Tiempo de consulta coseno con índices: ~7ms para 49,600 filas.

---

## 10. Pipeline de ingesta

```bash
# Ingesta normal (solo procesa documentos nuevos)
python ingest.py

# Re-ingesta completa desde cero
python ingest.py --reset

# Cambiar el rango SVD
python ingest.py --k 7

# Combinado
python ingest.py --reset --k 9
```

### Flags

| Flag | Default | Descripción |
|---|---|---|
| `--reset` | False | Trunca TERM, WORD, HAS, SVD_MATRIX y DOCUMENT antes de reingestar |
| `--k` | 100 | Rango SVD a computar (se clampea automáticamente a min(shape)-1) |

### Pasos del pipeline

```
[1/6] Conectando a MySQL...
[2/6] Cargando reglas de preprocesamiento...
[3/6] Extrayendo texto de PDFs...
[4/6] Construyendo matriz de frecuencias...
[5/6] Calculando SVD (k=9)...
[6/6] Guardando en base de datos...
Listo. 16878 términos, 10 documentos, 49600 celdas, SVD almacenado k=9.
```

### Idempotencia

El pipeline usa `INSERT IGNORE` (TERM, WORD, STOP_WORD, SUFFIX) y
`ON DUPLICATE KEY UPDATE` (HAS) para que ejecutarlo dos veces sin
`--reset` produzca exactamente el mismo resultado sin duplicar datos.

---

## 11. Motor de consulta y funciones de similitud

### Métodos disponibles

| Método | Parámetro | Tipo | Rango | Mejor para |
|---|---|---|---|---|
| `cosine` | `method='cosine'` | Similitud | [0, 1] | Lenguaje natural (recomendado) |
| `dice` | `method='dice'` | Similitud | [0, 1] | Overlap balanceado |
| `jaccard` | `method='jaccard'` | Similitud | [0, 1] | Penalizar baja coincidencia |
| `euclidean` | `method='euclidean'` | Distancia | [0, ∞) | Distancia espacial |
| `manhattan` | `method='manhattan'` | Distancia | [0, ∞) | Distancia city-block |

Para similitudes (coseno, Dice, Jaccard): mayor score = más relevante.
Para distancias (euclidiana, Manhattan): menor score = más relevante.

**Propiedad matemática garantizada para vectores no negativos:**
```
coseno(a,b) ≥ Dice(a,b) ≥ Jaccard(a,b)
```

Verificada por `test_cosine_gt_dice_gt_jaccard` en la suite de tests.

### Efecto del parámetro k

| k | Efecto |
|---|---|
| k bajo (2-3) | Solo temas dominantes; consultas específicas fallan |
| k medio (4-6) | ~80% varianza explicada; balance óptimo típico |
| k máximo (9) | Toda la varianza; máxima especificidad |

Con 10 documentos, k=9 es el máximo matemático (`min(M,N) - 1`).
Al agregar más documentos al corpus, k puede aumentar.

---

## 12. Capa LLM

### Jerarquía de selección automática

```
1. Ollama local en http://localhost:11434
   └── Selecciona modelo según RAM disponible
       ├── >= 15 GB → qwen2.5:7b
       ├── >= 7 GB  → llama3.2:3b
       └── < 7 GB   → phi3:mini
2. API de Google Gemini (si GEMINI_API_KEY está configurada)
3. Sin LLM — solo recuperación de documentos
```

### Instalar Ollama (opcional)

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar un modelo
ollama pull llama3.2:3b    # ~2 GB, funciona con 8 GB RAM
ollama pull phi3:mini       # ~1.6 GB, funciona con 4 GB RAM
ollama pull qwen2.5:7b      # ~4.4 GB, mejor español, requiere 16 GB RAM

# Iniciar servidor
ollama serve
```

### Selector manual en la UI

El sidebar muestra todos los modelos disponibles en un selectbox.
Seleccionar "Desactivado — solo recuperación" desactiva la síntesis
sin afectar la recuperación de documentos.

### Degradación elegante

Si el LLM falla (timeout, error de red, modelo no disponible):
- Los resultados de recuperación se muestran normalmente
- No aparece mensaje de error del LLM (solo se omite la sección)
- El sistema sigue siendo 100% funcional para recuperación

### Citación en la respuesta

El LLM siempre incluye al final de su respuesta:
```
Fuente principal: [Título del documento]
'[Cita textual del fragmento más relevante]'
```

---

## 13. Interfaz Streamlit

### Inicio rápido

```bash
streamlit run app.py
# http://localhost:8501
```

### Flujo de búsqueda

1. Escribir consulta → presionar Enter o botón Buscar
2. Barra de progreso: preprocesamiento → LSI → documentos encontrados → fragmentos → listo
3. Tarjetas de resultados con fragmento relevante y botón de descarga
4. Síntesis LLM con spinner animado (`st.status`)
5. Resultados persisten en `st.session_state` entre reruns

### Estructura de app.py

```python
# Orden de definición (importante para Streamlit):
st.set_page_config(...)          # Primera línea obligatoriamente
_render_results(...)             # Función auxiliar antes del layout
_run_query_with_timeout(...)     # Wrapper de timeout con threading
SearchTimeout                    # Excepción personalizada

# Sidebar: configuración + selector LLM
# Tab 1: Consulta (pipeline principal)
# Tab 2: Matriz FrecT
# Tab 3: Explorador SVD
# Tab 4: Documentos
```

### Caché de recursos

```python
@st.cache_data
def load_pdf_bytes(path: str) -> bytes:
    ...
```

Los PDFs se cargan una vez y se cachean por Streamlit para evitar
re-lecturas en cada render cuando top_n > 1.

---

## 14. Suite de pruebas

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Solo tests unitarios (sin DB)
pytest tests/test_preprocessor.py tests/test_matrix_builder.py \
       tests/test_svd_engine.py tests/test_similarity.py \
       tests/test_query_engine.py -v

# Solo tests de integración (requiere DB)
pytest tests/test_query_integration.py -v

# Reporte de cobertura
pytest tests/ --tb=short -q
```

### Estructura de la suite

| Archivo | Tests | Requiere DB | Cubre |
|---|---|---|---|
| `test_preprocessor.py` | 9 | No | tokenize, strip_suffix, preprocess |
| `test_matrix_builder.py` | 5 | No | build_frequency_matrix, apply_tfidf |
| `test_svd_engine.py` | 5 | No | compute_svd, project_query |
| `test_similarity.py` | 8 | No | fórmulas coseno, Dice, Jaccard, Euclid, Manhattan |
| `test_query_engine.py` | 7 | No | query(), cache, errores |
| `test_query_integration.py` | 7 | Sí | pipeline end-to-end, vistas SQL |

**Total: 53 tests — todos deben pasar antes de entregar.**

### Tests de integración con DB real

Los tests en `test_query_integration.py` se saltan automáticamente si
la base de datos no es alcanzable (usando `pytest.mark.skipif`).

Requieren corpus actual. Si cambias el corpus, actualizar:
- `test_known_query_ranking` — título esperado en top-3
- `test_all_methods_return_results` — query que produce resultados

---

## 15. Referencia de comandos

### Base de datos

```bash
# Crear schema desde cero
mysql -u usuario -p < schema.sql

# Conectar a la DB
mysql -u usuario -p docbase

# Ver similitud entre todos los pares (vista SQL)
mysql -u usuario -p docbase -e "SELECT * FROM v_cosine_similarity;"

# Ver normas de documentos
mysql -u usuario -p docbase -e "SELECT * FROM v_document_norms;"

# Contar registros por tabla
mysql -u usuario -p docbase -e "
  SELECT 'DOCUMENT' tbl, COUNT(*) n FROM DOCUMENT UNION
  SELECT 'TERM', COUNT(*) FROM TERM UNION
  SELECT 'HAS', COUNT(*) FROM HAS UNION
  SELECT 'SVD_MATRIX', COUNT(*) FROM SVD_MATRIX UNION
  SELECT 'STOP_WORD', COUNT(*) FROM STOP_WORD;"
```

### Diagnóstico rápido desde Python

```python
# Verificar tokens de una consulta
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from src.db import get_db
from src.preprocessor import load_stop_words, load_suffix_rules, preprocess_query

with get_db() as conn:
    sw = load_stop_words(conn)
    sr = load_suffix_rules(conn)
    tokens = preprocess_query("tu consulta aquí", sw, sr)
    print(f"Tokens: {tokens}")
EOF

# Verificar resultados de una consulta
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from src.db import get_db
from src.query_engine import query

with get_db() as conn:
    results = query("tu consulta aquí", conn, method='cosine', top_n=5)
    for r in results:
        print(f"[{r['score']:.4f}] {r['title']}")
EOF
```

---

## 16. Decisiones de diseño y limitaciones conocidas

### Por qué MySQL en lugar de una base de datos vectorial

El requisito académico del proyecto especifica usar un DBMS relacional
(Oracle, MySQL, PostgreSQL) con un esquema diseñado para representar la
matriz FrecT. Esto permite que las funciones de similitud se expresen
como SQL estándar ejecutable directamente en MySQL, lo que es el
objetivo pedagógico del curso.

Una base de datos vectorial (pgvector, Pinecone, Weaviate) ofrecería
mejor rendimiento en escala, pero no permitiría mostrar las consultas
SQL de similitud que el profesor evalúa.

### Limitación: homogeneidad del corpus

Con 10 documentos del mismo dominio jurídico laboral, el espacio LSI
a k=9 muestra compresión de scores (todos los documentos resultan
70-95% similares entre sí). Esto es una característica del corpus,
no un bug del sistema. Un corpus más diverso o con más documentos
mejoraría la discriminación.

Rango de similitud actual: 0.06 → 0.71 (considerado aceptable).
Par más similar: LIFNVT ↔ LSS (0.71) — ambas son leyes de seguridad
social con vocabulario muy similar.
Par menos similar: LISR ↔ NOM-035 (0.06) — ley fiscal vs norma de
riesgo psicosocial.

### Limitación: saturación en consultas cortas

Consultas de 1-2 términos muy frecuentes en el corpus producen scores
de 1.0 en múltiples documentos simultáneamente. El campo "Términos
coincidentes" actúa como desempate. Consultas de 3-5 términos
específicos producen rankings más discriminativos.

### Limitación: stemming rudimentario

El stemmer basado en reglas de sufijos es más simple que algoritmos
como Snowball/Porter. Esto puede producir sobre-stemming ocasional
(e.g., "accidente" → "accide" en lugar de "accident"). Para un sistema
de producción se recomendaría usar `nltk.stem.SnowballStemmer('spanish')`.

### Por qué Streamlit en lugar de una app de escritorio

Streamlit permite que el evaluador ejecute el sistema con un único
comando (`streamlit run app.py`) y lo visualice en cualquier navegador
sin instalaciones adicionales. Una app de escritorio (Electron, PyInstaller)
requeriría builds específicos por plataforma.

---

## 17. Solución de problemas

### Error: `ARPACK error -9: Starting vector is zero`

La matriz FrecT está vacía o tiene columnas nulas. Causas comunes:
- PDFs ilegibles en `data/pdfs/` (verificar con `pdfplumber`)
- Tabla `DOCUMENT` con registros obsoletos del corpus anterior

Solución:
```bash
python ingest.py --reset
```

### Error: `None of the query terms exist in the corpus`

Ningún token de la consulta existe en la tabla `TERM`. Diagnóstico:
```bash
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from src.db import get_db
from src.preprocessor import load_stop_words, load_suffix_rules, preprocess_query
with get_db() as conn:
    sw = load_stop_words(conn)
    sr = load_suffix_rules(conn)
    print(preprocess_query("tu consulta", sw, sr))
EOF
```

Si retorna lista vacía, la consulta contiene solo palabras vacías o
términos no indexados. Usar terminología más específica del dominio.

### La síntesis LLM no aparece

Verificar que Ollama está corriendo:
```bash
curl http://localhost:11434/api/tags
# Si falla: ollama serve &
```

O configurar `GEMINI_API_KEY` en `.env` como alternativa.

### Scores todos en 1.0

Consulta demasiado corta o términos muy frecuentes. Solución: agregar
más términos específicos a la consulta. Ver recomendaciones en sección 6.

### La aplicación no carga

```bash
# Verificar MySQL
mysql -u usuario -p -e "USE docbase; SHOW TABLES;"

# Verificar .env
cat .env

# Verificar importaciones
python -c "import app"

# Ver logs completos
streamlit run app.py --logger.level=debug
```

### Tests fallan después de cambiar el corpus

Los tests de integración en `test_query_integration.py` pueden
hardcodear títulos del corpus anterior. Actualizar:
- `test_known_query_ranking`: cambiar título esperado
- `test_all_methods_return_results`: cambiar query de prueba

---

*Documentación generada para DocBase — LIS-3012 UDLAP*
