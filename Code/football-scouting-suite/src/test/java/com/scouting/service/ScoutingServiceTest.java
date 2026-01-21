package com.scouting.service;

import com.scouting.data.model.Player;
import com.scouting.data.repository.PlayerRepository;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Arrays;
import java.util.List;
import static org.mockito.Mockito.when;

/**
 * Questo è un classico "Unit Test" isolato con Mockito.
 * Qui testiamo la logica del Service *senza* coinvolgere il database reale. Usiamo un "Mock" (una finta implementazione / Stub)
 * del Repository per simulare i dati.
 * Vogliamo verificare che il Service faccia il suo lavoro (es. chiamare findAll) assumendo che il DB funzioni.
 * Questo rende il test veloce e focalizzato esclusivamente sulla logica Java del servizio.
 */

@ExtendWith(MockitoExtension.class) // Abilita Mockito
class ScoutingServiceTest {

    @Mock
    private PlayerRepository playerRepository;

    @InjectMocks
    private ScoutingService scoutingService;

    @Test
    void testGetAllPlayers() {
        Player p1 = new Player();
        p1.setName("Player A");
        Player p2 = new Player();
        p2.setName("Player B");
        
        when(playerRepository.findAll()).thenReturn(Arrays.asList(p1, p2));

        List<Player> result = scoutingService.getAllPlayers();

        Assertions.assertEquals(2, result.size());
        Assertions.assertEquals("Player A", result.get(0).getName());
    }
}