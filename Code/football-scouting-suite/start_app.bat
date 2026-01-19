@echo off
cls
echo ========================================================
echo   AVVIO SCOUTING APP
echo   Il browser si aprira' automaticamente tra pochi secondi
echo ========================================================

:: 1. Controllo presenza Java
java -version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo [ERRORE] Java non e' installato o non e' nel PATH.
    echo Installa Java 17 o superiore.
    pause
    exit /b
)

:: 2. Strategia di esecuzione Maven
if exist "mvnw.cmd" (
    echo [INFO] Trovato Maven Wrapper. Avvio in corso...
    call mvnw.cmd spring-boot:run
) else (
    echo [INFO] Maven Wrapper 'mvnw.cmd' non trovato.
    echo [INFO] Tentativo di utilizzo di Maven globale...
    
    where mvn >nul 2>&1
    if %errorlevel% EQU 0 (
        echo [INFO] Maven globale trovato. Avvio in corso...
        call mvn spring-boot:run
    ) else (
        echo.
        echo [ERRORE FATALE]
        echo Non e' stato possibile avviare l'applicazione perche':
        echo 1. Il file 'mvnw.cmd' non esiste nella cartella.
        echo 2. Maven non risulta installato globalmente nel sistema.
        echo.
        echo Soluzione: Installa Apache Maven oppure genera il wrapper da Eclipse.
        pause
        exit /b
    )
)

pause