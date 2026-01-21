package com.scouting.data.model;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Questo è il primo test, focalizzato sull'Entità.
 * Verifica che i getter ed i setter funzionino come previsto. Anche se sembra banale, in progetti grandi è utile
 * per assicurarsi che nessuno rompa l'incapsulamento o introduca una logica sbagliata nei metodi di accesso ai dati.
 * Garantisce l'integrità dell'oggetto base del nostro dominio.
 */

class PlayerTest {

    @Test
    void testPlayerAttributes() {
        // Setup
        Player player = new Player();
        Long expectedId = 1L;
        String expectedName = "Messi";
        Double expectedGoals = 0.95;

        player.setId(expectedId);
        player.setName(expectedName);
        player.setGoalsP90(expectedGoals);

        Assertions.assertEquals(expectedId, player.getId());
        Assertions.assertEquals(expectedName, player.getName());
        Assertions.assertEquals(expectedGoals, player.getGoalsP90());
    }
}