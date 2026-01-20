package com.scouting;

import com.scouting.UI.MainView;
import com.scouting.service.ScoutingService;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;

@SpringBootTest // Carica l'intero contesto Spring (Database H2, Service, UI, ecc.)
class ScoutingApplicationTests {

    @Autowired
    private ScoutingService scoutingService;

    @MockBean
    private MainView mainView;

    @Test
    void contextLoads() {
        // Verifica che i componenti principali siano stati creati in memoria
        Assertions.assertNotNull(scoutingService, "Il Service dovrebbe essere caricato nel contesto");
        Assertions.assertNotNull(mainView, "La View principale dovrebbe essere caricata nel contesto");
    }
}