# ⚽ Football Scouting Suite

**Football Scouting Suite** è un'applicazione per lo scouting di calciatori, basata sui dati della stagione **2025-26** (Database Agg. 31/12/2025).
Il sistema integra una pipeline **ETL** (Extract, Transform, Load) automatizzata, un motore di ricerca dinamico basato su specifiche formali e un'interfaccia web reattiva.

---

## 🏗️ Architettura del Software

Il progetto è progettato seguendo rigorosi principi di Ingegneria del Software:

- **Design Pattern MVC (Model-View-Controller):** Netta separazione tra UI (Vaadin), Logica di Business (Service) e Dati (Repository/Entity).
- **Strategy Pattern:** Utilizzato nella `PlayerSpecificationFactory` per gestire dinamicamente filtri su tipi di dati eterogenei (Double, Integer) rispettando l'Open/Closed Principle.
- **Repository Pattern:** Astrazione dell'accesso ai dati tramite Spring Data JPA.
- **ETL Pipeline:** Un servizio dedicato (`DataProcessingService`) gestisce l'importazione, la pulizia e la normalizzazione dei dati CSV raw.

---

## 🚀 Guida all'Installazione e Avvio

### Prerequisiti
- **Java JDK 17** o superiore installato.
- **Maven** (opzionale).
- **Python 3.x** (solo se si desidera rieseguire lo scraping dei dati).

### 1. Clonazione del Progetto
```bash
git clone https://github.com/AndreaBrivio/ScoutingSuite.git
cd ScoutingSuite/Code/football-scouting-suite
```

### 2. Avvio Rapido (Windows)
Se preferisci usare la riga di comando:

```bash
.\start_app.bat
```

### 3. Avvio Manuale (Windows)
Nella cartella principale del progetto è presente uno script di avvio automatico.
Fai doppio click su:

`start_app.bat`

Questo script verifica l'ambiente, compila il progetto, avvia il server e apre automaticamente il browser.


Una volta avviato, l'applicazione sarà accessibile a: http://localhost:8080

---

## 🧪 Quality Assurance & Testing
Il progetto integra strumenti avanzati per l'analisi statica e dinamica del codice.
Eseguire i Test Unitari e di Integrazione (JUnit 5)

- **Per lanciare la suite completa dei test (inclusi i test con Database H2 in-memory e Mockito):**

```bash
mvn test
```

- **Analisi Statica del Codice (PMD):**

```bash
mvn pmd:check
```

- **Genera un report HTML completo in target/site/pmd.html:**

```bash
mvn pmd:pmd
```

- **Analisi delle Metriche e Dipendenze (JDepend) con report HTML completo in target/site/jdepend-report.html:**

```bash
mvn site
```

- **Generazione Documentazione (Javadoc), navigabile in target/site/apidocs/index.html:**

```bash
mvn javadoc:javadoc
```

---

## 🐍 Aggiornamento Dati (Python Scraper)
I dati dei giocatori sono estratti da FBref tramite uno script Python che utilizza Selenium.
Se necessiti di dati aggiornati:
Assicurati di avere Python e Chrome installati.
Installa le dipendenze:

```bash
cd ScoutingSuite/Code/football-scouting-suite\src\main\resources
pip install pandas selenium beautifulsoup4 webdriver-manager lxml
```

Esegui lo script:
```bash
python Scraping_and_Cleaning.py
```
Il nuovo file Player_Final.csv verrà salvato nella cartella delle risorse e caricato automaticamente al prossimo riavvio dell'applicazione Java.

---

## 📂 Struttura del Progetto

```bash
src.main
├── java.com.scouting
│   ├── config
│   ├── data
│   │   ├── model
│   │   └── repository
│   ├── service
│   │   ├── specification
│   │       └── strategy
│   └── ui
└── resourses
    └── Season 2025-26
```

---

## 🛠️ Tecnologie Utilizzate

- **Backend**: Java 17, Spring Boot 3.1, Spring Data JPA
- **Frontend**: Vaadin Flow 24 (Java-based UI framework)
- **Database**: H2 Database (In-Memory per alta velocità)
- **Tools**: Maven, Lombok, Tablesaw (CSV parsing)
- **Testing**: JUnit 5, Mockito, Spring Boot Test
- **Quality**: SonarQube, PMD, JDepend

---

## 📄 Licenza
Questo progetto è rilasciato sotto licenza MIT.
