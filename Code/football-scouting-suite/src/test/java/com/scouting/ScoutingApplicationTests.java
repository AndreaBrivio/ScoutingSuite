package com.scouting;

import com.scouting.service.ScoutingService;
import com.scouting.ui.MainView;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;

/**
 * Questo è un "Smoke Test" (Test di Fumo).
 * Il suo scopo non è verificare una logica specifica, ma assicurarsi che l'applicazione riesca almeno ad avviarsi.
 * Carica l'intero contesto di Spring (Database, Controller, Service) e verifica che i componenti principali (Bean)
 * siano stati creati correttamente. Se questo fallisce, c'è un problema grave di configurazione.
 */

@SpringBootTest
class ScoutingApplicationTests {

    @Autowired
    private ScoutingService scoutingService;

    @MockBean
    private MainView mainView;

    @Test
    void contextLoads() {
        Assertions.assertNotNull(scoutingService, "Il Service dovrebbe essere caricato nel contesto");
        Assertions.assertNotNull(mainView, "La View principale dovrebbe essere caricata nel contesto");
    }
}